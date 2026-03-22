import os
from typing import List

import gradio as gr
from groq import Groq
from pypdf import PdfReader
from langdetect import detect, LangDetectException
from langdetect import detect_langs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are **GrindZone AI**, an educational assistant named Alex.

About you:
- You are 20 years old, a university student who also works part-time.
- You are friendly, patient, and encouraging.
- You communicate clearly and efficiently.

Core Behavior Rules:
1. **Do NOT assume missing information**
   - If the user request lacks important details, ASK clarifying questions BEFORE giving a full answer.
   - Keep questions concise and grouped.

2. **Be concise by default**
   - Avoid long paragraphs unless explicitly needed.
   - Prefer short explanations, bullet points, and structured responses.

3. **Adapt response depth**
   - If the user gives minimal input → ask questions first.
   - If the user gives detailed input → provide a full, high-quality answer.


Your capabilities:

1. **Explain concepts**
   - Break down complex topics simply.
   - Use examples when helpful.

2. **Summarise notes**
   - Start with a short overview.
   - Then key bullet points.
   - End with study tips.

3. **Create study plans (IMPORTANT BEHAVIOR)**
   - BEFORE creating a study plan, you MUST check for:
     • Course / subject
     • Deadline or exam date
     • Current level (beginner/intermediate/etc.)
     • Weekly availability (hours per day/week)
     • Goals (pass, high grade, mastery)

   - If ANY of these are missing:
     → Ask the user for them FIRST (in a clean bullet list)
     → DO NOT generate the plan yet

   - Only generate the study plan AFTER you have enough information.

Communication style:
- Use bullet points and clean formatting
- Keep responses structured and easy to scan
- Be supportive but not overly verbose

Internationalisation:
- ALWAYS detect the language of the user's message and respond entirely in THAT language.
- Default to English only if uncertain.
"""

ALLOWED_EXTENSIONS = {".txt", ".pdf"}

# ---------------------------------------------------------------------------
# UI Translations
# ---------------------------------------------------------------------------

UI_LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Español": "es",
}

UI_TRANSLATIONS = {
    "en": {
        "header_title": "📚 GrindZone AI 🔐",
        "header_subtitle": "Your friendly AI study companion 😊 Ask questions, upload notes, build study plans",
        "chat_label": "Chat with your Lock In Buddy",
        "chat_placeholder": "👋 Hi! I'm Alex, your Lock In Buddy.\nAsk me to explain a concept, summarise your notes, or help you create a study plan!",
        "msg_placeholder": "Type your question here…",
        "send_btn": "Send ➤",
        "settings_title": "### ⚙️ Settings",
        "lang_label": "🌐 UI Language",
        "file_label": "📎 Upload notes (.txt / .pdf)",
        "font_label": "🔤 Chat font size",
        "clear_btn": "🗑️ Clear conversation",
        "tips": "\n**Tips:**\n- Upload **.txt** or **.pdf** files and hit send to get a summary.\n- Ask in **any language**, I'll respond in your language!\n- Try: *\"Create a 2-week study plan for calculus\"*",
        "btn_study": "📅 Create Study Plan",
        "btn_explain": "📖 Explain a Concept",
        "btn_summary": "📝 Summarise Notes",
        "prompt_study": "Create a 2-week study plan for my course.",
        "prompt_explain": "Explain this concept in simple terms:",
        "prompt_summary": "Please summarise my uploaded notes.",
    },
    "fr": {
        "header_title": "📚 GrindZone AI",
        "header_subtitle": "Votre compagnon d'étude IA 😊 Posez des questions, téléversez vos notes, créez des plans d'étude",
        "chat_label": "Discuter avec votre Lock In Buddy",
        "chat_placeholder": "👋 Salut ! Je suis Alex, votre Lock In Buddy.\nDemandez-moi d'expliquer un concept, résumer vos notes ou créer un plan d'étude !",
        "msg_placeholder": "Écrivez votre question ici…",
        "send_btn": "Envoyer ➤",
        "settings_title": "### ⚙️ Paramètres",
        "lang_label": "🌐 Langue de l'interface",
        "file_label": "📎 Téléverser des notes (.txt / .pdf)",
        "font_label": "🔤 Taille de police du chat",
        "clear_btn": "🗑️ Effacer la conversation",
        "tips": "\n**Astuces :**\n- Téléversez des fichiers **.txt** ou **.pdf** et appuyez sur Envoyer pour obtenir un résumé.\n- Écrivez dans **n'importe quelle langue**, je répondrai dans votre langue !\n- Essayez : *« Crée un plan d'étude de 2 semaines pour le calcul »*",
        "btn_study": "📅 Créer un plan d'étude",
        "btn_explain": "📖 Expliquer un concept",
        "btn_summary": "📝 Résumer les notes",
        "prompt_study": "Crée un plan d'étude de 2 semaines pour mon cours.",
        "prompt_explain": "Explique ce concept simplement :",
        "prompt_summary": "Veuillez résumer mes notes téléchargées.",
    },
    "es": {
        "header_title": "📚 GrindZone AI",
        "header_subtitle": "Tu compañero de estudio IA 😊 Haz preguntas, sube apuntes, crea planes de estudio",
        "chat_label": "Chatea con tu Lock In Buddy",
        "chat_placeholder": "👋 ¡Hola! Soy Alex, tu Lock In Buddy.\n¡Pídeme que explique un concepto, resuma tus apuntes o cree un plan de estudio!",
        "msg_placeholder": "Escribe tu pregunta aquí…",
        "send_btn": "Enviar ➤",
        "settings_title": "### ⚙️ Configuración",
        "lang_label": "🌐 Idioma de la interfaz",
        "file_label": "📎 Subir apuntes (.txt / .pdf)",
        "font_label": "🔤 Tamaño de fuente del chat",
        "clear_btn": "🗑️ Borrar conversación",
        "tips": "\n**Consejos:**\n- Sube archivos **.txt** o **.pdf** y presiona Enviar para obtener un resumen.\n- Escribe en **cualquier idioma**, ¡responderé en tu idioma!\n- Prueba: *\"Crea un plan de estudio de 2 semanas para cálculo\"*",
        "btn_study": "📅 Crear plan de estudio",
        "btn_explain": "📖 Explicar un concepto",
        "btn_summary": "📝 Resumir apuntes",
        "prompt_study": "Crea un plan de estudio de 2 semanas para mi curso.",
        "prompt_explain": "Explica este concepto de forma sencilla:",
        "prompt_summary": "Por favor resume mis apuntes subidos.",
    },
}

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Please export it before running the app:\n"
                "  export GROQ_API_KEY='your-key-here'"
            )
        _client = Groq(api_key=api_key)
    return _client

# ---------------------------------------------------------------------------
# Lang detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Detect the language of the user message.
    Rules:
    - Default to English for short messages (< 4 words)
    - Otherwise, detect language and only switch if confidence >= 70%
    """
    if len(text.split()) < 4:
        return "en"  # short messages like 'Hi', 'Hello', 'Hola'
    try:
        langs = detect_langs(text)
        if langs and langs[0].prob >= 0.7:
            return langs[0].lang
        return "en"  # low confidence fallback
    except LangDetectException:
        return "en"

# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def _extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages_text)

    raise ValueError(
        f"Unsupported file type: '{ext}'. "
        "Please upload a **.txt** or **.pdf** file."
    )

# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

def _build_messages(
    history: List[dict],
    user_message: str,
    file_context: str | None = None,
) -> List[dict]:
    detected = detect_language(user_message)
    lang_hint = (
        f"\n\n[System note: The user appears to be writing in language code '{detected}'. "
        f"Please respond entirely in that language.]"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT + lang_hint}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    if file_context:
        messages.append(
            {
                "role": "user",
                "content": (
                    "I've uploaded a document. Here is its content:\n\n"
                    f"```\n{file_context[:12000]}\n```\n\n"
                    "Please summarise or use this in your answer."
                ),
            }
        )

    messages.append({"role": "user", "content": user_message})
    return messages


def respond(user_message: str, history: List[dict], uploaded_file: str | None):
    if not user_message.strip() and not uploaded_file:
        yield history
        return

    file_context: str | None = None
    if uploaded_file is not None:
        try:
            file_context = _extract_text_from_file(uploaded_file)
        except ValueError as exc:
            history.append({"role": "assistant", "content": f"⚠️ {exc}"})
            yield history
            return

    if not user_message.strip():
        user_message = "Please summarise the uploaded document."

    history.append({"role": "user", "content": user_message})

    messages = _build_messages(history[:-1], user_message, file_context)

    try:
        stream = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=2048,
            stream=True,
        )
    except EnvironmentError as exc:
        history.append({"role": "assistant", "content": f"⚠️ {exc}"})
        yield history
        return
    except Exception as exc:
        history.append(
            {"role": "assistant", "content": f"⚠️ Something went wrong: {exc}"}
        )
        yield history
        return

    partial = ""
    history.append({"role": "assistant", "content": ""})
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            partial += delta
            history[-1]["content"] = partial
            yield history

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --sb-primary:  #6C63FF;
    --sb-primary-dark: #5A52D5;
    --sb-accent:   #FF6584;
    --sb-bg:       #0F1117;
    --sb-surface:  #1A1D2E;
    --sb-surface2: #252839;
    --sb-text:     #E8E8F0;
    --sb-muted:    #9395A5;
    --sb-border:   #2E3148;
    --sb-radius:   12px;
}

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: var(--sb-bg) !important;
    color: var(--sb-text) !important;
}

#header-row {
    background: linear-gradient(135deg, var(--sb-primary), #8B5CF6, var(--sb-accent));
    border-radius: var(--sb-radius);
    padding: 28px 32px;
    margin-bottom: 8px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.25);
}
#header-row h1 {
    margin: 0; font-size: 2rem; font-weight: 700; color: #fff; letter-spacing: -0.5px;
}
#header-row p {
    margin: 6px 0 0; color: rgba(255,255,255,0.85); font-size: 1rem;
}

#chatbot {
    border: 1px solid var(--sb-border) !important;
    border-radius: var(--sb-radius) !important;
    background: var(--sb-surface) !important;
    min-height: 480px;
}
#chatbot .message { border-radius: 10px !important; }
#chatbot .user {
    background: linear-gradient(135deg, var(--sb-primary), #8B5CF6) !important;
    color: #fff !important;
}
#chatbot .bot {
    background: var(--sb-surface2) !important;
    color: var(--sb-text) !important;
}

#msg-box textarea {
    background: var(--sb-surface) !important;
    border: 1px solid var(--sb-border) !important;
    border-radius: var(--sb-radius) !important;
    color: var(--sb-text) !important;
    font-family: 'Inter', sans-serif !important;
    padding: 14px 16px !important;
}
#msg-box textarea:focus {
    border-color: var(--sb-primary) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.2) !important;
}

.gr-button-primary, #send-btn {
    background: linear-gradient(135deg, var(--sb-primary), #8B5CF6) !important;
    border: none !important; border-radius: 10px !important;
    color: #fff !important; font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.gr-button-primary:hover, #send-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(108,99,255,0.35) !important;
}

#clear-btn {
    background: var(--sb-surface2) !important;
    border: 1px solid var(--sb-border) !important;
    border-radius: 10px !important;
    color: var(--sb-text) !important;  /* brighter text */
    font-weight: 500 !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}
#clear-btn:hover {
    background: var(--sb-surface) !important;
    color: var(--sb-text) !important;
}

#file-upload {
    border: 2px dashed var(--sb-border) !important;
    border-radius: var(--sb-radius) !important;
    background: var(--sb-surface) !important;
    transition: border-color 0.2s ease !important;
}
#file-upload:hover { border-color: var(--sb-primary) !important; }

#font-slider input[type=range] { accent-color: var(--sb-primary) !important; }

#settings-col {
    background: var(--sb-surface) !important;
    border: 1px solid var(--sb-border) !important;
    border-radius: var(--sb-radius) !important;
    padding: 20px !important;
}

#chatbot.font-sm .message,
#chatbot.font-sm .message *,
#chatbot.font-sm .user,
#chatbot.font-sm .user *,
#chatbot.font-sm .bot,
#chatbot.font-sm .bot * {
  font-size: 14px !important;
  line-height: 1.55 !important;
}

#chatbot.font-md .message,
#chatbot.font-md .message *,
#chatbot.font-md .user,
#chatbot.font-md .user *,
#chatbot.font-md .bot,
#chatbot.font-md .bot * {
  font-size: 16px !important;
  line-height: 1.6 !important;
}

#chatbot.font-lg .message,
#chatbot.font-lg .message *,
#chatbot.font-lg .user,
#chatbot.font-lg .user *,
#chatbot.font-lg .bot,
#chatbot.font-lg .bot * {
  font-size: 18px !important;
  line-height: 1.65 !important;
}

#chatbot.font-xl .message,
#chatbot.font-xl .message *,
#chatbot.font-xl .user,
#chatbot.font-xl .user *,
#chatbot.font-xl .bot,
#chatbot.font-xl .bot * {
  font-size: 20px !important;
  line-height: 1.7 !important;
}
"""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def fill_prompt(prompt_text):
    return gr.update(value=prompt_text)

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="GrindZone AI",
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="pink",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        # ---- Header ----
        with gr.Row(elem_id="header-row"):
            header_html = gr.HTML(
                f"""
                <h1>{UI_TRANSLATIONS["en"]["header_title"]}</h1>
                <p>{UI_TRANSLATIONS["en"]["header_subtitle"]}</p>
                """
            )

        # ---- Main layout ----
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label=UI_TRANSLATIONS["en"]["chat_label"],
                    height=520,
                    placeholder=UI_TRANSLATIONS["en"]["chat_placeholder"],
                )

                with gr.Row():
                    btn_study = gr.Button(UI_TRANSLATIONS["en"]["btn_study"])
                    btn_explain = gr.Button(UI_TRANSLATIONS["en"]["btn_explain"])
                    btn_summary = gr.Button(UI_TRANSLATIONS["en"]["btn_summary"])


                with gr.Row():
                    msg = gr.Textbox(
                        elem_id="msg-box",
                        placeholder=UI_TRANSLATIONS["en"]["msg_placeholder"],
                        lines=2,
                        scale=5,
                        show_label=False,
                    )
                    send_btn = gr.Button(
                        UI_TRANSLATIONS["en"]["send_btn"],
                        elem_id="send-btn",
                        variant="primary",
                        scale=1,
                        min_width=100,
                    )

            with gr.Column(scale=1, elem_id="settings-col", min_width=240):
                settings_md = gr.Markdown(UI_TRANSLATIONS["en"]["settings_title"])

                lang_dropdown = gr.Dropdown(
                    choices=list(UI_LANGUAGES.keys()),
                    value="English",
                    label=UI_TRANSLATIONS["en"]["lang_label"],
                    elem_id="lang-dropdown",
                )

                font_slider = gr.Slider(
                    elem_id="font-slider",
                    minimum=14,
                    maximum=22,
                    step=2,
                    value=16,
                    label=UI_TRANSLATIONS["en"]["font_label"],
                )

                clear_btn = gr.Button(
                    UI_TRANSLATIONS["en"]["clear_btn"],
                    elem_id="clear-btn",
                )

                gr.Markdown("---") 

                file_upload = gr.File(
                    elem_id="file-upload",
                    label=UI_TRANSLATIONS["en"]["file_label"],
                    file_types=[".txt", ".pdf"],
                    type="filepath",
                )

                tips_md = gr.Markdown(UI_TRANSLATIONS["en"]["tips"])


        # ----------------------------------------------------------------
        # Language switcher callback
        # ----------------------------------------------------------------
        def change_language(lang_name):
            code = UI_LANGUAGES.get(lang_name, "en")
            t = UI_TRANSLATIONS.get(code, UI_TRANSLATIONS["en"])
            return (
                # header_html
                f'<h1>{t["header_title"]}</h1><p>{t["header_subtitle"]}</p>',
                # chatbot label
                gr.update(label=t["chat_label"], placeholder=t["chat_placeholder"]),
                # msg placeholder
                gr.update(placeholder=t["msg_placeholder"]),
                # send_btn
                gr.update(value=t["send_btn"]),
                # settings_md
                t["settings_title"],
                # lang_dropdown label
                gr.update(label=t["lang_label"]),
                # file_upload label
                gr.update(label=t["file_label"]),
                # font_slider label
                gr.update(label=t["font_label"]),
                # clear_btn
                gr.update(value=t["clear_btn"]),
                # tips_md
                t["tips"],
                # btn_study
                gr.update(value=t["btn_study"]),
                # btn_explain
                gr.update(value=t["btn_explain"]),
                # btn_summary
                gr.update(value=t["btn_summary"]),
            )

        lang_dropdown.change(
            fn=change_language,
            inputs=[lang_dropdown],
            outputs=[
                header_html,
                chatbot,
                msg,
                send_btn,
                settings_md,
                lang_dropdown,
                file_upload,
                font_slider,
                clear_btn,
                tips_md,
                btn_study,
                btn_explain,
                btn_summary,
            ],
        )

        # ---- Chat event handlers ----
        def on_submit(user_message, history, uploaded_file):
            for updated_history in respond(user_message, history or [], uploaded_file):
                yield updated_history, gr.update(value=""), None

        send_btn.click(
            fn=on_submit,
            inputs=[msg, chatbot, file_upload],
            outputs=[chatbot, msg, file_upload],
        )

        def fill_prompt_by_lang(lang_name, key):
            code = UI_LANGUAGES.get(lang_name, "en")
            t = UI_TRANSLATIONS.get(code, UI_TRANSLATIONS["en"])
            return t[key]

        btn_study.click(
            fn=lambda lang: fill_prompt_by_lang(lang, "prompt_study"),
            inputs=lang_dropdown,
            outputs=msg,
        )

        btn_explain.click(
            fn=lambda lang: fill_prompt_by_lang(lang, "prompt_explain"),
            inputs=lang_dropdown,
            outputs=msg,
        )

        btn_summary.click(
            fn=lambda lang: fill_prompt_by_lang(lang, "prompt_summary"),
            inputs=lang_dropdown,
            outputs=msg,
        )
        
        msg.submit(
            fn=on_submit,
            inputs=[msg, chatbot, file_upload],
            outputs=[chatbot, msg, file_upload],
        )

        clear_btn.click(
            fn=lambda: ([], None),
            outputs=[chatbot, file_upload],
        )

        font_slider.change(
            fn=None,
            inputs=[font_slider],
            js="""
            (size) => {
                let chatbot = document.querySelector("#chatbot");
                if (!chatbot) return;
                chatbot.classList.remove("font-sm","font-md","font-lg","font-xl");
                if (size <= 14) chatbot.classList.add("font-sm");
                else if (size <= 16) chatbot.classList.add("font-md");
                else if (size <= 18) chatbot.classList.add("font-lg");
                else chatbot.classList.add("font-xl");
                return [];
            }
            """,
        )

    return demo

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )