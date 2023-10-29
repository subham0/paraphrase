from transformers import PegasusForConditionalGeneration, PegasusTokenizer, T5Tokenizer, T5ForConditionalGeneration

class ParaphrasingModels:
    def __init__(self):
        self.tokenizers = {}
        self.models = {}
        self.load_models()

    def load_models(self):
        # Loading the PEGASUS model for paraphrasing
        model_name_pegasus = "tuner007/pegasus_paraphrase"
        self.tokenizers["tuner007/pegasus_paraphrase"] = PegasusTokenizer.from_pretrained(model_name_pegasus)
        self.models["tuner007/pegasus_paraphrase"] = PegasusForConditionalGeneration.from_pretrained(model_name_pegasus)

        # Loading Vamsi/T5_Paraphrase_Paws model
        model_name_t5 = "Vamsi/T5_Paraphrase_Paws"
        self.tokenizers["Vamsi/T5_Paraphrase_Paws"] = T5Tokenizer.from_pretrained(model_name_t5)
        self.models["Vamsi/T5_Paraphrase_Paws"] = T5ForConditionalGeneration.from_pretrained(model_name_t5)

    def paraphrase_text(self, text, model_name, num_paraphrases):
        paraphrases = []
        if model_name == "tuner007/pegasus_paraphrase":
            input_ids = self.tokenizers[model_name].encode(text, return_tensors="pt")
            outputs = self.models[model_name].generate(
                input_ids, 
                max_length=100, 
                num_return_sequences=num_paraphrases, 
                no_repeat_ngram_size=2, 
                early_stopping=True, 
                temperature=0.99,
                num_beams=max(5, num_paraphrases)
            )
            for output in outputs:
                paraphrase = self.tokenizers[model_name].decode(output, skip_special_tokens=True)
                paraphrases.append(paraphrase)
        elif model_name == "Vamsi/T5_Paraphrase_Paws":
            input_text = "paraphrase: " + text
            input_ids = self.tokenizers[model_name].encode(input_text, return_tensors="pt")
            outputs = self.models[model_name].generate(
                input_ids, 
                max_length=100, 
                num_return_sequences=num_paraphrases, 
                early_stopping=True, 
                temperature=0.99, 
                num_beams=max(5, num_paraphrases),
                no_repeat_ngram_size=2
            )
            for output in outputs:
                paraphrase = self.tokenizers[model_name].decode(output, skip_special_tokens=True)
                paraphrases.append(paraphrase)
        return paraphrases
