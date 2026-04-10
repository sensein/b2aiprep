from gliner import GLiNER
from transformers import AutoTokenizer, AutoModelForCausalLM
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

import torch

class Models:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymiser = AnonymizerEngine()

    def prompt_phi4(self, prompt):
        print(type(prompt))
        model_id = "microsoft/Phi-4-mini-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        llm_output = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return llm_output

    def prompt_presidio(self, prompt):
        analyzer_results = self.analyzer.analyze(text=prompt, language="en")
        results = ""
        if analyzer_results:
            results = self.anonymiser.anonymize(text=prompt, analyzer_results=analyzer_results)
        return analyzer_results ,results

    def prompt_gliner(self, prompt, labels):
        model = GLiNER.from_pretrained("nvidia/gliner-pii")
        model_output = model.predict_entities(prompt, labels, threshold=0.5)
        return model_output