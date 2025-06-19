import json, torch, gradio as gr
from transformers import pipeline

torch.set_num_threads(1) #to facilitate running on 1-cpu specially for HuggingSpace 
print("Loading NLLB-200 translator …")
translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    device=-1,                      # CPU
    torch_dtype=torch.float32,
)
with open("language.json", encoding="utf-8") as f:
    lang_tbl = {row["Language"].lower(): row["FLORES-200 code"]
                for row in json.load(f)}
langs = sorted(lang_tbl.keys())



def translate_text(text: str, dst_lang: str) -> str:
    """English text → chosen language."""
    if not text.strip():
        return "⚠️ Please enter some text."
    code = lang_tbl.get(dst_lang.lower())
    if code is None:
        return "⚠️ Language not found."
    out = translator(text,
                     src_lang="eng_Latn",
                     tgt_lang=code,
                     max_length=512)
    return out[0]["translation_text"]


#UI
gr.close_all()

with gr.Blocks(title="🌐 English → 200-language Translator (CPU)") as demo:
    gr.Markdown(
        "## 🌐 English → 200-language Translator (CPU-only)\n"
        "Type your English text and get an instant translation."
    )

    with gr.Row():
        text_input  = gr.Textbox(lines=4, label="English text")
        lang_drop   = gr.Dropdown(langs, value="german", label="Translate to")

    text_output = gr.Textbox(lines=4, label="Translation")

    gr.Button("Translate").click(
        fn=translate_text,
        inputs=[text_input, lang_drop],
        outputs=text_output
    )

demo.launch()
