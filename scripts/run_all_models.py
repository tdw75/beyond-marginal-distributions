import os
import sys

import fire

print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.prompting.messages import Survey
from src.simulation.experiment import (
    generate_run_id,
    huggingface_login,
    load_experiment,
)
from src.simulation.models import adapters, ModelConfig, load_model, change_subgroup
from src.simulation.inference import run_single
from src.simulation.survey import load_survey, save_results


def main(
    experiment_name: str,
    run_id: str = None,
    root_directory: str = "",
    **kwargs,  # additional LLM hyperparams
):
    experiment = load_experiment(experiment_name, root_directory)

    run_id = run_id or generate_run_id(experiment.simulation["base_model_name"])
    simulated_surveys = {}

    shared_config_vars = {**experiment.simulation, "hyperparams": kwargs}

    # todo: clear cache in loop

    survey_questions = load_survey(experiment, "individual", False)
    survey_flipped = load_survey(experiment, "individual", True)
    opinion_gpt_surveys = run_opinion_gpt(
        survey_questions, survey_flipped, shared_config_vars, run_id
    )
    phi_instruct_surveys = run_phi_instruct(
        survey_questions, survey_flipped, shared_config_vars, run_id
    )
    simulated_surveys.update(phi_instruct_surveys)
    simulated_surveys.update(opinion_gpt_surveys)

    save_results(
        simulated_surveys, experiment.files["directory"], experiment.setup["name"]
    )


def run_phi_instruct(
    survey_questions: Survey,
    survey_flipped: Survey,
    shared_config_vars: dict,
    run_id: str,
):
    simulated_surveys = {}
    config = ModelConfig(
        **shared_config_vars,
        is_lora=False,
        is_persona=True,
        aggregation_by="questions",
    )
    model, tokenizer = load_model(config)

    for subgroup in adapters + [None]:
        model, config = change_subgroup(model, config, subgroup)
        simulated_surveys[config.run_name] = run_single(
            model, tokenizer, config, survey_questions, survey_flipped, run_id
        )
    return simulated_surveys


def run_opinion_gpt(
    survey_questions: Survey,
    survey_flipped: Survey,
    shared_config_vars: dict,
    run_id: str,
):
    # todo: add persona prompting for opinion gpt
    simulated_surveys = {}
    config = ModelConfig(
        **shared_config_vars,
        is_lora=True,
        is_persona=False,
        aggregation_by="questions",
    )
    model, tokenizer = load_model(config)

    for subgroup in adapters:
        model, config = change_subgroup(model, config, subgroup)
        simulated_surveys[config.run_name] = run_single(
            model, tokenizer, config, survey_questions, survey_flipped, run_id
        )
    return simulated_surveys


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
