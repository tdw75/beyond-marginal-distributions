import logging
import re
from typing import Literal, Any

from peft import PeftModel
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from src.prompting.system import (
    build_survey_context_message,
    build_survey_context_for_persona,
)

logger = logging.getLogger(__name__)

ModelName = str
AdapterName = str

bias_to_subreddit: dict[AdapterName, str] = {
    "liberal": "AskALiberal",
    "conservative": "AskConservatives",
    "german": "AskAGerman",
    "american": "AskAnAmerican",
    "latin_america": "AskLatinAmerica",
    "middle_east": "AskMiddleEast",
    "men": "AskMen",
    "women": "AskWomen",
    "people_over_30": "AskPeopleOver30",
    "old_people": "AskOldPeople",
    # "teenager": "AskTeenagers",
}

adapters: list[AdapterName] = list(bias_to_subreddit.keys())


def replace_adapter_name(adapter: AdapterName, base_model_name: str):
    """Convert adapter name to the correct format for the given base model."""
    if base_model_name == "llama" and adapter == "latin_america":
        return "latin_american"
    else:
        return adapter


MODEL_DIRECTORY = {
    "phi": "unsloth/Phi-3-mini-4k-instruct",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    # "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}


class ModelConfig(BaseModel):
    base_model_name: Literal[tuple(MODEL_DIRECTORY.keys())] = "phi"
    subgroup: Literal[tuple(adapters + [None])] = None
    is_lora: bool = False
    is_persona: bool = False
    device: str = "cuda:0"
    aggregation_by: Literal["questions", "respondent"] = "questions"
    decoding_style: Literal["constrained", "unconstrained"] = "unconstrained"
    sample_size: int = 500
    batch_size: int = 50
    hyperparams: dict = {}
    system_prompt: str = None

    def model_post_init(self, context: Any, /) -> None:
        default_hyperparams: dict[str, Any] = dict(
            max_new_tokens=16, top_p=0.9, temperature=0.6, do_sample=True
        )
        self.hyperparams = {**default_hyperparams, **self.hyperparams}
        self.system_prompt = self.system_prompt or build_survey_context_message()

    @property
    def model_type(self) -> str:
        return "opinion-gpt" if self.is_lora else f"instruct"

    @property
    def model_id(self) -> str:
        return MODEL_DIRECTORY.get(self.base_model_name, self.base_model_name)

    @property
    def is_phi_model(self):
        return bool(re.match(r"^.+/phi.*", self.model_id, re.IGNORECASE))

    @property
    def is_llama_model(self):
        return bool(re.match(r"^.+/llama.*", self.model_id, re.IGNORECASE))

    def change_subgroup(self, subgroup: str):
        if subgroup in adapters + [None]:
            self.subgroup = subgroup
        else:
            raise ValueError(f"Invalid subgroup: {subgroup}")

    @property
    def run_name(self):
        persona_suffix = "with-persona" if self.is_persona else "no-persona"
        return f"{self.base_model_name}-{self.model_type}-{self.subgroup or 'general'}-{persona_suffix}"


def load_model(
    config: ModelConfig,
) -> tuple[PeftModel | PreTrainedModel, PreTrainedTokenizer]:
    model, tokenizer = load_base(config)
    if config.is_lora:
        return load_opinion_gpt(model, config), tokenizer
    else:
        logger.info("No LoRA adapters used")
        return model.to(config.device), tokenizer


def load_opinion_gpt(model: PreTrainedModel, config: ModelConfig) -> PeftModel:

    lora_id = _get_lora_id(config.base_model_name)
    default_adapter = adapters[0]

    model = PeftModel.from_pretrained(
        model, lora_id.format(adapter=default_adapter), adapter_name=default_adapter
    ).to(config.device)

    for adapter in adapters[1:]:  # all adapters loaded to be accessed as needed
        adapter_name = replace_adapter_name(adapter, config.base_model_name)
        logger.info(f"Loading adapter: {adapter_name}")
        model.load_adapter(lora_id.format(adapter=adapter_name), adapter_name)

    return model


def _get_lora_id(base_model_name: str) -> str:
    if base_model_name == "phi":
        return "HU-Berlin-ML-Internal/opiniongpt-phi3-{adapter}"
    elif base_model_name == "llama":
        return "Opinion-GPT/llama2-{adapter}"
    else:
        raise ValueError(f"Unknown base model name: {base_model_name}")


def load_base(config: ModelConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, padding_side="left")
    if config.is_llama_model:
        tokenizer.chat_template = LLAMA2_TOKENIZER_TEMPLATE
    # if is_phi_model(model_id):
    #     tokenizer.chat_template = PHI_TOKENIZER_FORMAT
    logger.info(f"Successfully loaded model: {config.model_id}")
    return model, tokenizer


def change_subgroup(
    model: PreTrainedModel | PeftModel, config: ModelConfig, new_subgroup: str
) -> tuple[PreTrainedModel | PeftModel, ModelConfig]:
    config.change_subgroup(new_subgroup)
    if config.is_lora:
        adapter_name = replace_adapter_name(config.subgroup, config.base_model_name)
        model = change_adapter(model, adapter_name)
    if config.is_persona:
        config.system_prompt = build_survey_context_for_persona(config.subgroup)
    return model, config


def change_adapter(model: PeftModel, target_adapter: str) -> PeftModel:
    if model.active_adapter != target_adapter:
        model.set_adapter(target_adapter)
        logger.info(f"Changed active adapter to {target_adapter}")
    return model


PHI_TOKENIZER_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] + eos_token }}"
    "\n{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)


LLAMA2_TOKENIZER_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'system' %}"
    "{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + message['content'] + ' ' + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)
