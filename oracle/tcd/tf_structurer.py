import numpy as np
import subprocess
import tempfile
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, NamedTuple

logger = logging.getLogger(__name__)


class PocketInfo(NamedTuple):
    pocket_id: int
    center: np.ndarray
    volume: float
    druggability_score: float
    hydrophobicity: float
    residues: List[int]
    pocket_type: str


class TFStructureResult(NamedTuple):
    tf_name: str
    pdb_path: str
    chain_id: str
    residues: List[Dict]
    domains: Dict[str, Tuple[int, int]]
    best_pocket: PocketInfo
    all_pockets: List[PocketInfo]
    perturbation_type: str
    structure_source: str


class TFStructurer:
    """Prepares protein structures for TCIP warhead design.

    Steps: retrieve structure (PDB > ESMFold > AlphaFold) → parse PDB →
    annotate domains → run fpocket → run MD ensemble → detect cryptic pockets
    → select optimal binding pocket.
    """

    def __init__(
        self,
        pdb_dir: Path = Path("./data/raw/pdb"),
        alphafold_dir: Path = Path("./data/raw/alphafold"),
        md_frames: int = 100,
        md_step_ps: float = 10.0,
        fpocket_executable: str = "fpocket",
    ):
        self.pdb_dir = pdb_dir
        self.alphafold_dir = alphafold_dir
        self.md_frames = md_frames
        self.md_step_ps = md_step_ps
        self.fpocket_exe = fpocket_executable

        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        self.alphafold_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self, tf_name: str, perturbation_type: str) -> TFStructureResult:
        logger.info(f"Preparing structure for {tf_name} ({perturbation_type})")

        pdb_path, source = self._get_structure(tf_name)
        logger.info(f"  Structure source: {source} ({pdb_path})")

        residues, chain_id = self._parse_pdb(pdb_path)
        logger.info(f"  Structure: {len(residues)} residues, chain {chain_id}")

        domains = self._annotate_domains(tf_name, residues)
        logger.info(f"  Domains: {list(domains.keys())}")

        crystal_pockets = self._run_fpocket(pdb_path)
        logger.info(f"  fpocket: {len(crystal_pockets)} pockets found")

        try:
            ensemble_pdb_paths = self._run_md_ensemble(pdb_path)
            cryptic_pockets = self._detect_cryptic_pockets(crystal_pockets, ensemble_pdb_paths)
            logger.info(f"  Cryptic pockets: {len(cryptic_pockets)} found")
        except Exception as e:
            logger.warning(f"  MD/cryptic pocket detection failed: {e}")
            cryptic_pockets = []

        all_pockets = crystal_pockets + cryptic_pockets

        best_pocket = self._select_best_pocket(all_pockets, domains, perturbation_type)
        logger.info(
            f"  Best pocket: volume={best_pocket.volume:.1f}Å³, "
            f"drug_score={best_pocket.druggability_score:.3f}, "
            f"type={best_pocket.pocket_type}"
        )

        return TFStructureResult(
            tf_name=tf_name,
            pdb_path=str(pdb_path),
            chain_id=chain_id,
            residues=residues,
            domains=domains,
            best_pocket=best_pocket,
            all_pockets=all_pockets,
            perturbation_type=perturbation_type,
            structure_source=source,
        )

    def _get_structure(self, tf_name: str) -> Tuple[Path, str]:
        for suffix, source in [("_experimental.pdb", "pdb"), ("_esm.pdb", "esm"), ("_af.pdb", "alphafold")]:
            local = self.pdb_dir / f"{tf_name}{suffix}"
            if local.exists():
                return local, source

        pdb_path = self._fetch_from_pdb(tf_name)
        if pdb_path is not None:
            return pdb_path, "pdb"

        esm_path = self._fetch_from_esm(tf_name)
        if esm_path is not None:
            return esm_path, "esm"

        af_path = self._fetch_from_alphafold(tf_name)
        if af_path is not None:
            return af_path, "alphafold"

        raise ValueError(f"No structure available for {tf_name}")

    def _fetch_from_pdb(self, tf_name: str) -> Optional[Path]:
        import requests
        try:
            search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
            query = {
                "query": {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_gene_name.value",
                        "operator": "exact_match",
                        "value": tf_name,
                    },
                },
                "return_type": "entry",
                "request_options": {
                    "paginate": {"start": 0, "rows": 5},
                    "sort": [{"sort_by": "score", "direction": "desc"}],
                },
            }
            resp = requests.post(search_url, json=query, timeout=10)
            if resp.status_code != 200:
                return None
            results = resp.json().get("result_set", [])
            if not results:
                return None
            pdb_id = results[0]["identifier"]
            pdb_path = self.pdb_dir / f"{tf_name}_experimental.pdb"
            download_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            resp2 = requests.get(download_url, timeout=30)
            if resp2.status_code == 200:
                pdb_path.write_bytes(resp2.content)
                logger.info(f"  Downloaded PDB {pdb_id} for {tf_name}")
                return pdb_path
        except Exception as e:
            logger.debug(f"PDB fetch failed for {tf_name}: {e}")
        return None

    def _fetch_from_esm(self, tf_name: str) -> Optional[Path]:
        import requests
        sequence = self._get_uniprot_sequence(tf_name)
        if sequence is None:
            return None
        try:
            resp = requests.post(
                "https://api.esmatlas.com/foldSequence/v1/pdb/",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=sequence,
                timeout=120,
            )
            if resp.status_code == 200:
                pdb_path = self.pdb_dir / f"{tf_name}_esm.pdb"
                pdb_path.write_bytes(resp.content)
                logger.info(f"  ESMFold structure obtained for {tf_name}")
                return pdb_path
        except Exception as e:
            logger.debug(f"ESMFold failed for {tf_name}: {e}")
        return None

    def _fetch_from_alphafold(self, tf_name: str) -> Optional[Path]:
        import requests
        uniprot_id = self._gene_to_uniprot(tf_name)
        if uniprot_id is None:
            return None
        try:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                af_path = self.alphafold_dir / f"{tf_name}_af.pdb"
                af_path.write_bytes(resp.content)
                logger.info(f"  AlphaFold structure downloaded for {tf_name}")
                return af_path
        except Exception as e:
            logger.debug(f"AlphaFold fetch failed for {tf_name}: {e}")
        return None

    def _parse_pdb(self, pdb_path: Path) -> Tuple[List[Dict], str]:
        from Bio.PDB import PDBParser
        import warnings
        from Bio.PDB.PDBExceptions import PDBConstructionWarning

        parser = PDBParser(QUIET=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PDBConstructionWarning)
            structure = parser.get_structure("protein", str(pdb_path))

        model = structure[0]
        chain = list(model.get_chains())[0]
        chain_id = chain.id

        residues = []
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            ca_coord = None
            if "CA" in res:
                ca_coord = np.array(res["CA"].get_vector().get_array())
            residues.append({
                "name": res.resname,
                "id": res.id[1],
                "ca_coord": ca_coord,
            })

        return residues, chain_id

    def _annotate_domains(self, tf_name: str, residues: List[Dict]) -> Dict[str, Tuple[int, int]]:
        KNOWN_DOMAINS: Dict[str, Dict[str, Tuple[int, int]]] = {
            "CDX2": {"DBD": (146, 213), "transactivation": (1, 145)},
            "SNAI2": {"ZF1": (153, 175), "ZF2": (178, 200), "ZF3": (203, 225), "ZF4": (228, 250), "SNAG": (1, 20)},
            "MYC": {"bHLH": (368, 439), "LZ": (440, 463), "MBD": (1, 100)},
            "GATA3": {"ZF1": (261, 294), "ZF2": (319, 352), "TA": (1, 100)},
            "CEBPA": {"bZIP": (290, 358), "TA1": (1, 100), "TA2": (130, 200)},
            "MITF": {"bHLH": (180, 250), "LZ": (251, 290), "TA": (1, 50)},
        }
        if tf_name in KNOWN_DOMAINS:
            return KNOWN_DOMAINS[tf_name]

        n_res = len(residues)
        return {
            "N_terminal": (1, n_res // 3),
            "middle": (n_res // 3, 2 * n_res // 3),
            "C_terminal": (2 * n_res // 3, n_res),
        }

    def _run_fpocket(self, pdb_path: Path) -> List[PocketInfo]:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "fpocket_out"
            try:
                result = subprocess.run(
                    [self.fpocket_exe, "-f", str(pdb_path), "-o", str(output_dir)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except FileNotFoundError:
                logger.warning("fpocket not found. Using simplified pocket detection.")
                return self._simplified_pocket_detection(pdb_path)

            pockets = self._parse_fpocket_output(output_dir, pdb_path)
        return pockets

    def _simplified_pocket_detection(self, pdb_path: Path) -> List[PocketInfo]:
        residues, _ = self._parse_pdb(pdb_path)
        coords = np.array([r["ca_coord"] for r in residues if r["ca_coord"] is not None])
        if len(coords) == 0:
            return []
        center = coords.mean(axis=0)
        return [PocketInfo(
            pocket_id=0,
            center=center,
            volume=500.0,
            druggability_score=0.5,
            hydrophobicity=0.5,
            residues=list(range(min(20, len(residues)))),
            pocket_type="crystal",
        )]

    def _parse_fpocket_output(self, output_dir: Path, pdb_path: Path) -> List[PocketInfo]:
        pockets = []
        summary_file = output_dir / f"{pdb_path.stem}_info.txt"
        if not summary_file.exists():
            info_files = list(output_dir.glob("*_info.txt"))
            if not info_files:
                return self._simplified_pocket_detection(pdb_path)
            summary_file = info_files[0]

        try:
            with open(summary_file) as f:
                content = f.read()

            pocket_blocks = re.split(r"Pocket\s+\d+", content)[1:]
            for i, block in enumerate(pocket_blocks):
                score_match = re.search(r"Druggability Score\s*:\s*([\d.]+)", block)
                volume_match = re.search(r"Volume\s*:\s*([\d.]+)", block)
                hydrophob_match = re.search(r"Hydrophobicity Score\s*:\s*([\d.]+)", block)

                drug_score = float(score_match.group(1)) if score_match else 0.3
                volume = float(volume_match.group(1)) if volume_match else 300.0
                hydrophob = float(hydrophob_match.group(1)) if hydrophob_match else 0.5

                pockets.append(PocketInfo(
                    pocket_id=i,
                    center=np.zeros(3),
                    volume=volume,
                    druggability_score=drug_score,
                    hydrophobicity=hydrophob,
                    residues=[],
                    pocket_type="crystal",
                ))

        except Exception as e:
            logger.warning(f"fpocket parsing failed: {e}")
            return self._simplified_pocket_detection(pdb_path)

        pockets.sort(key=lambda p: p.druggability_score, reverse=True)
        return pockets

    def _run_md_ensemble(self, pdb_path: Path) -> List[Path]:
        try:
            from openmm import app, unit, Platform
            from openmm import LangevinMiddleIntegrator
        except ImportError:
            logger.warning("OpenMM not available. Skipping MD ensemble.")
            return []

        logger.info("  Running MD ensemble sampling (this takes a few minutes)...")
        try:
            forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
            pdb = app.PDBFile(str(pdb_path))
            modeller = app.Modeller(pdb.topology, pdb.positions)
            modeller.addHydrogens(forcefield)
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
                implicitSolvent=app.OBC2,
                soluteDielectric=1.0,
                solventDielectric=78.5,
            )
            integrator = LangevinMiddleIntegrator(
                310.0 * unit.kelvin,
                1.0 / unit.picoseconds,
                2.0 * unit.femtoseconds,
            )

            try:
                platform = Platform.getPlatformByName("OpenCL")
            except Exception:
                platform = Platform.getPlatformByName("CPU")

            simulation = app.Simulation(modeller.topology, system, integrator, platform)
            simulation.context.setPositions(modeller.positions)
            simulation.minimizeEnergy(maxIterations=500)
            simulation.context.setVelocitiesToTemperature(310.0 * unit.kelvin)
            simulation.step(5000)

            frame_paths = []
            steps_per_frame = int(self.md_step_ps * 1000)
            for frame_i in range(self.md_frames):
                simulation.step(steps_per_frame)
                state = simulation.context.getState(getPositions=True)
                positions = state.getPositions(asNumpy=True)
                frame_path = pdb_path.parent / f"{pdb_path.stem}_frame_{frame_i:04d}.pdb"
                with open(frame_path, "w") as f:
                    app.PDBFile.writeFile(modeller.topology, positions, f)
                frame_paths.append(frame_path)

            logger.info(f"  MD ensemble: {len(frame_paths)} frames")
            return frame_paths
        except Exception as e:
            logger.warning(f"MD simulation failed: {e}")
            return []

    def _detect_cryptic_pockets(
        self, crystal_pockets: List[PocketInfo], ensemble_paths: List[Path]
    ) -> List[PocketInfo]:
        cryptic = []
        if not ensemble_paths:
            return cryptic

        all_frame_pockets = []
        for frame_path in ensemble_paths[:20]:
            try:
                frame_pockets = self._run_fpocket(frame_path)
                all_frame_pockets.extend(frame_pockets)
            except Exception:
                continue

        if not all_frame_pockets:
            return cryptic

        centers = np.array([
            p.center for p in all_frame_pockets
            if not np.all(p.center == 0)
        ])
        if len(centers) < 3:
            return cryptic

        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=4.0, min_samples=3).fit(centers)
        labels = clustering.labels_

        for label in set(labels):
            if label < 0:
                continue
            cluster_pockets = [all_frame_pockets[i] for i, l in enumerate(labels) if l == label]
            frequency = len(cluster_pockets) / len(ensemble_paths)
            if frequency < 0.2:
                continue

            mean_center = np.mean([p.center for p in cluster_pockets], axis=0)
            is_new = all(
                np.linalg.norm(mean_center - cp.center) > 6.0
                for cp in crystal_pockets
            )
            if is_new:
                mean_vol = np.mean([p.volume for p in cluster_pockets])
                mean_drug = np.mean([p.druggability_score for p in cluster_pockets])
                cryptic.append(PocketInfo(
                    pocket_id=len(crystal_pockets) + len(cryptic),
                    center=mean_center,
                    volume=mean_vol,
                    druggability_score=mean_drug,
                    hydrophobicity=0.5,
                    residues=[],
                    pocket_type="cryptic",
                ))

        return cryptic

    def _select_best_pocket(
        self,
        pockets: List[PocketInfo],
        domains: Dict[str, Tuple[int, int]],
        perturbation_type: str,
    ) -> PocketInfo:
        if not pockets:
            return PocketInfo(0, np.zeros(3), 400.0, 0.5, 0.5, [], "crystal")

        scored = []
        for pocket in pockets:
            score = 0.0
            score += 0.4 * pocket.druggability_score

            v = pocket.volume
            if 300 <= v <= 800:
                vol_score = 1.0
            elif v < 300:
                vol_score = v / 300.0
            else:
                vol_score = max(0, 1.0 - (v - 800) / 1000.0)
            score += 0.3 * vol_score

            if perturbation_type == "activate" and "DBD" in domains:
                dbd_start, dbd_end = domains["DBD"]
                pocket_in_dbd = any(dbd_start <= r <= dbd_end for r in pocket.residues)
                if pocket_in_dbd:
                    score -= 0.2

            if pocket.pocket_type == "cryptic":
                score += 0.1

            scored.append((score, pocket))

        scored.sort(reverse=True)
        return scored[0][1]

    def _get_uniprot_sequence(self, gene_name: str) -> Optional[str]:
        import requests
        try:
            uniprot_id = self._gene_to_uniprot(gene_name)
            if uniprot_id is None:
                return None
            resp = requests.get(
                f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta",
                timeout=10,
            )
            if resp.status_code == 200:
                lines = resp.text.splitlines()
                return "".join(lines[1:])
        except Exception:
            pass
        return None

    def _gene_to_uniprot(self, gene_name: str) -> Optional[str]:
        import requests
        try:
            resp = requests.get(
                f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}+AND+organism_id:9606+AND+reviewed:true&format=tsv&fields=accession",
                timeout=10,
            )
            if resp.status_code == 200:
                lines = resp.text.strip().splitlines()
                if len(lines) > 1:
                    return lines[1].strip()
        except Exception:
            pass
        return None
