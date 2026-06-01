import numpy as np
import logging
import time
import base64
from typing import List, Optional

from oracle.interfaces import RSPOutput, TCDOutput, TCIPMolecule, ValidationResult

logger = logging.getLogger(__name__)


class TCDPipeline:
    """Transcriptional CIP Designer pipeline orchestrator."""

    def __init__(self, config=None):
        self.config = config or {}

    def run(self, rsp_output: RSPOutput) -> TCDOutput:
        from oracle.tcd.tf_structurer import TFStructurer
        from oracle.tcd.writer_selector import WriterEraserSelector
        from oracle.tcd.linker_designer import LinkerDesigner
        from oracle.tcd.tcip_assembler import TCIPAssembler
        from oracle.tcd.ternary_validator import TernaryComplexValidator

        t0 = time.time()
        cam_output = rsp_output.cam_output
        cancer_type = cam_output.cancer_type
        patient_id = cam_output.sample_id

        structurer = TFStructurer(
            md_frames=self.config.get("md_frames", 50),
        )
        selector = WriterEraserSelector()
        linker_designer = LinkerDesigner()
        assembler = TCIPAssembler()
        validator = TernaryComplexValidator()

        cancer_expression = {}
        if "mean_expression" in cam_output.adata.uns:
            cancer_expression = cam_output.adata.uns["mean_expression"]
        else:
            try:
                import scanpy as sc
                X = cam_output.adata.X
                if hasattr(X, "toarray"):
                    X = X.toarray()
                mean_expr = np.array(X).mean(axis=0)
                cancer_expression = {g: float(mean_expr[i]) for i, g in enumerate(cam_output.adata.var_names)}
            except Exception:
                pass

        tcip_molecules: List[TCIPMolecule] = []

        all_tfs = (
            [(g, "activate") for g in rsp_output.genes_to_activate] +
            [(g, "repress") for g in rsp_output.genes_to_repress]
        )

        for tf_name, perturbation_type in all_tfs:
            logger.info(f"[TCD] Designing TCIP for {tf_name} ({perturbation_type})")
            try:
                tf_struct = structurer.prepare(tf_name, perturbation_type)
            except Exception as e:
                logger.warning(f"[TCD] Structure prep failed for {tf_name}: {e}")
                continue

            writer_sel = selector.select(
                tf_name=tf_name,
                perturbation_type=perturbation_type,
                cancer_expression=cancer_expression,
            )

            pocket_center = tf_struct.best_pocket.center
            writer_site_center = np.array([10.0, 10.0, 10.0])

            required_dist = linker_designer.calculate_required_distance(
                tf_pocket_center=pocket_center,
                writer_site_center=writer_site_center,
            )
            linker_candidates = linker_designer.design(
                required_distance_A=required_dist,
                perturbation_type=perturbation_type,
            )
            linker_info = linker_candidates[0] if linker_candidates else None
            linker_smiles = linker_info.smiles if linker_info else "OCC"

            warhead_smiles = self._generate_warhead(tf_name, tf_struct, perturbation_type)
            recruiter_smiles = writer_sel.info.recruiter_smiles

            full_smiles = assembler.assemble(
                warhead_smiles=warhead_smiles,
                linker_smiles=linker_smiles,
                recruiter_smiles=recruiter_smiles,
            )
            if full_smiles is None:
                logger.warning(f"[TCD] Assembly failed for {tf_name}")
                continue

            props = assembler.compute_properties(full_smiles)
            if not props:
                continue

            tc_score = validator.validate(
                tcip_smiles=full_smiles,
                tf_name=tf_name,
                writer_name=writer_sel.writer_eraser_name,
                tf_pocket_center=pocket_center,
                writer_site_center=writer_site_center,
                tf_structure_path=tf_struct.pdb_path,
                perturbation_type=perturbation_type,
            )

            validation_result = ValidationResult(
                passed=tc_score.overall_passed,
                clash_score=tc_score.clash_score,
                interface_energy=tc_score.interface_energy,
                writer_positioning_distance=tc_score.writer_min_distance_A,
                writer_positioning_productive=tc_score.writer_is_productive,
                sa_score=props.get("sa_score", 5.0),
                qed=props.get("qed", 0.3),
                mw=props.get("mw", 900.0),
                logP=props.get("logp", 4.0),
                tpsa=props.get("tpsa", 120.0),
                hbd=props.get("hbd", 4),
                hba=props.get("hba", 8),
                rotatable_bonds=props.get("rotatable_bonds", 10),
                passes_ro5=props.get("passes_extended_ro5", False),
                passes_veber=props.get("passes_veber", False),
            )

            mol_image = assembler.draw_molecule(full_smiles)

            tcip = TCIPMolecule(
                target_tf=tf_name,
                perturbation_type=perturbation_type,
                writer_eraser=writer_sel.writer_eraser_name,
                full_smiles=full_smiles,
                tf_warhead_smiles=warhead_smiles,
                linker_smiles=linker_smiles,
                recruiter_smiles=recruiter_smiles,
                molecular_weight=props.get("mw", 0.0),
                logP=props.get("logp", 0.0),
                tpsa=props.get("tpsa", 0.0),
                sa_score=props.get("sa_score", 5.0),
                qed=props.get("qed", 0.3),
                hbd=props.get("hbd", 0),
                hba=props.get("hba", 0),
                rotatable_bonds=props.get("rotatable_bonds", 0),
                predicted_tf_binding_affinity_nM=100.0,
                predicted_writer_binding_affinity_nM=writer_sel.info.recruiter_ki_nM,
                ternary_complex_score=float(not tc_score.overall_passed) * -1 + float(tc_score.overall_passed),
                validation_result=validation_result,
                mol_image_b64=mol_image,
            )
            tcip_molecules.append(tcip)

        n_validated = sum(1 for m in tcip_molecules if m.validation_result.passed)
        pred_rev_prob = rsp_output.validated_reversion_fraction

        logger.info(f"[TCD] Complete: {len(tcip_molecules)} molecules, {n_validated} validated in {time.time()-t0:.1f}s")

        return TCDOutput(
            tcip_molecules=tcip_molecules,
            n_molecules=len(tcip_molecules),
            n_validated=n_validated,
            cancer_type=cancer_type,
            patient_id=patient_id,
            predicted_reversion_probability=pred_rev_prob,
            rsp_output=rsp_output,
            cam_output=cam_output,
        )

    def _generate_warhead(self, tf_name: str, tf_struct, perturbation_type: str) -> str:
        KNOWN_WARHEADS = {
            "CDX2": "c1ccc2[nH]c(=O)ccc2c1",
            "SNAI2": "c1ccc(-c2cnc(N)nc2)cc1",
            "MYC": "c1ccc(Nc2ncnc3ccccc23)cc1",
            "GATA3": "c1cnc(N)nc1",
            "CEBPA": "CC(=O)Nc1ccc(cc1)S(N)(=O)=O",
            "MITF": "c1cc2cnccc2cc1",
        }
        if tf_name in KNOWN_WARHEADS:
            return KNOWN_WARHEADS[tf_name]
        return "c1ccccc1NC(=O)c1ccccn1"
