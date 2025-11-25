import streamlit as st
import requests
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import re
from zoneinfo import ZoneInfo

# --- CONFIGURACIÓN ---
# MODEL_NAME = "llama3:latest"
# MODEL_NAME = "llama3.2:3b"
MODEL_NAME = "llama3.2:1b"
# MODEL_NAME = "deepseek-r1"
# MODEL_NAME = "Mistral"
# MODEL_NAME = "gpt-oss"


# requests.post("http://localhost:11434/api/chat", json={
#     "model": MODEL_NAME,
#     "messages": [{"role": "user", "content": "ping"}]
# })

OLLAMA_HOST = "http://localhost:11434"
FORM_STATE_PREFIX = "FORM_STATE:"
DATA_PATH = Path("data/travel_forms.json")
DEPARTMENTS_PATH = Path("data/departments.json")
TODAY = datetime.utcnow().strftime("%d-%m-%Y")
today_weekday = datetime.utcnow().strftime("%A")

SYSTEM_PROMPT_TEMPLATE = ( 
"""
Eres un asistente inteligente responsable de guiar a los usuarios de la Universidad de Oviedo en la cumplimentación de solicitudes de Comisión de Servicio.
Estas solicitudes se usan cuando un miembro del personal debe viajar por motivos académicos (congresos, reuniones, estancias, etc.).
La fecha actual es {today} ({today_weekday}). Las fechas de salida y regreso deben ser posteriores a {today}. Comprueba internamente que las fechas sean válidas. Si la fecha de regreso es anterior a la de salida, informa al usuario.
Siempre responde con fechas en formato DD-MM-YYYY. Las horas deben ir en formato HH:MM.

Necesitas obtener los siguientes campos:

- nombre completo de la persona (name)
- identificador oficial (DNI, NIF, etc.) (id)
- departamento al que pertenece (department)
- grupo de investigación o unidad (group)
- motivo del viaje (congreso, reunión, estancia…) (cause)
- fecha de inicio de la actividad (start_activity)
- fecha de fin de la actividad (end_activity)
- ciudad o lugar de origen (origin)
- ciudad o lugar de destino (destination)
- fecha y hora de salida del viaje (departure_date y departure_time respectivamente)
- fecha y hora de regreso del viaje (return_date y return_time respectivamente)
# - fuente de financiación (proyecto, convenio, etc.) (commission)
- tipo de comisión (ej. Viajes y dietas) (commission)
- comentarios generales (comments)
- tipo de dieta aplicable (diet)
- gastos de alojamiento (importe estimado) (accommodation_expenses)
- comentarios sobre alojamiento (accommodation_comments)
- gastos de manutención (importe estimado) (maintenance_expenses)
- comentarios sobre manutención (maintenance_comments)
- gastos de locomoción (importe estimado) (locomotion_expenses)
- número de kilómetros (n_km)
- matrícula del vehículo (car_registration)
- comentarios sobre locomoción (locomotion_comments)
- gastos de inscripción (importe estimado) (registration_expenses)
- número de asistencias (n_attendance)
- gastos por asistencia (importe estimado) (attendance_expenses)
- comentarios sobre asistencia (attendance_comments)
- otros gastos (desglose por tipo e importe) (other_expenses)
- gastos gestionados por agencia de alojamiento (puede haber más de uno, con id (accommodation_agency_id), referencia presupuestaria (accommodation_budget_ref) y cantidad (accommodation_amount))
- gastos gestionados por agencia de locomocion (puede haber más de uno, con id (locomotion_agency_id), referencia presupuestaria (locomotion_budget_ref) y cantidad (locomotion_amount))
- datos del proyecto (referencia interna (internal_ref) y código de proyecto (project_code))
- resumen confirmado (sí/no) (resume)
- estado de autorización (pendiente/aprobada/rechazada) (authorization)
- estado del expediente (abierto/cerrado) (status)
- notas adicionales (notes)

Haz preguntas de seguimiento breves y de una en una hasta completar todos los datos.
Usa formato de fecha DD-MM-YYYY.
Comprueba que la fecha de regreso es posterior a la de salida y ambas son posteriores a la fecha actual.

Siempre incluye una línea final que empiece por FORM_STATE: seguida del JSON con el estado actual del formulario, aunque esté incompleto. Esta línea es solo para uso interno del sistema y no debe mostrarse como parte de la conversación con el usuario.

Al final de cada respuesta **debes** incluir UNA LÍNEA EXACTA que empiece por 'FORM_STATE:' seguida únicamente por un JSON válido (entre llaves) con el estado actual del formulario. Esa línea es obligatoria y no debe contener texto adicional.

Responde con mensajes naturales y breves que guíen la conversación. El sistema usará el FORM_STATE para completar datos automáticamente y continuar el flujo.

Nunca muestres razonamientos internos, ni análisis, ni pasos de pensamiento, ni uses etiquetas como <think> o similares. Solo responde con el mensaje final destinado al usuario.


El JSON inicial debe tener esta estructura:


FORM_STATE: {{
    "id": null,
    "name": null,
    "department": null,
    "group": null,
    "cause": null,
    "start_activity": null,
    "end_activity": null,
    "origin": null,
    "destination": null,
    "departure_date": null,
    "departure_time": null,
    "return_date": null,
    "return_time": null,
    "commission": null,
    "diet": null,
    "accommodation_expenses": null,
    "accommodation_comments": null,
    "maintenance_expenses": null,
    "maintenance_comments": null,
    "locomotion_expenses": null,
    "n_km": null,
    "car_registration": null,
    "locomotion_comments": null,
    "registration_expenses": null,
    "n_attendance": null,
    "attendance_expenses": null,
    "attendance_comments": null,
    "other_expenses": null,
    "accommodation_agency": {{
        "accommodation_agency_id": null,
        "accommodation_budget_ref": null,
        "accommodation_amount": null
    }},
    "locomotion_agency": {{
        "locomotion_agency_id": null,
        "locomotion_budget_ref": null,
        "locomotion_amount": null
    }},
    "project": {{
        "internal_ref": null,
        "project_code": null
    }},
    "comments": null,
    "resume": null,
    "authorization": null,
    "status": null,
    "notes": null
}}


Inicializa todas las variables como null

Procedimientos a seguir:

- Lo primero que tienes que hacer es pedir el DNI ('id').

- Los campos 'name' y 'department' no debes rellenarlos.

- Una vez tengas esto debes pedir los campos 'group' y 'cause'.

- Después tienes que pedir la fecha de inicio ('start_activity') y fin de la actividad ('end_activity'). En ningún caso la fecha de fin puede ser anterior a la fecha de inicio.


Restricciones y opciones válidas para campos específicos:

- Las fechas de inicio de la actividad ('start_activity') no podrá ser en ningún caso posterior a la fehca de fin de la actividad ('end_activity').

- Las fechas de salida del viaje ('departure_date') no podrá ser en ningún caso posterior a la fehca de regreso del viaje('return_date').

- Las fechas del viaje no pueden exceder de un día antes y un día después a las fechas de la actividad.

- El campo 'diet' debe ser uno de los siguientes valores. No se permite ningún otro. Indícale estos tres valores al usuario para que te indiquen cual elige:
    - 'Grupo II Personal con dietas minoradas del Ministerio'
    - 'Grupo II Dietas Universidad de Oviedo'
    - 'Grupo II Ciudades +210 mil habitantes'

- El campo 'commission' debe ser uno de los siguientes valores. Indícale estos cinco valores al usuario para que te indiquen cual elige:
    - 'Viajes y dietas'
    - 'Jornadas de seguimiento AEI'
    - 'Estancias Breves (1 a 3 meses). Sin limitación de número'
    - 'Estancias Breves (máximo 1 mes y una estancia durante el proyecto)'
    - 'Estancias Breves (1 a 3 meses por año)'

- El campo 'departure_time' y 'return_time' deben estar en formato HH:MM (24 horas). Si el usuario no lo indica, pregunta por ello.

- El campo 'locomotion_expenses' debe ir acompañado de:
    - 'n_km': número de kilómetros recorridos
    - 'car_registration': matrícula del vehículo
    - 'locomotion_comments': observaciones si las hubiera

- Las opciones válidas para 'locomotion_comments' deben referirse a uno de los siguientes medios:
    - Vehículo propio (requiere matrícula y kilómetros)
    - Tren
    - Autobús
    - Avión
    - Taxi
    - Coche de alquiler
    En caso de no ser vehículo propio los campos 'n_km' y 'car_registration' deben quedarse vacíos. Si va en vehívulo propio estos campos deben estar obligatoriamente rellenos.

- El campo 'comments' y todos los campos de tipo 'xxx_comments' deben rellenarse solo si proceden. Si el usuario no tiene observaciones, deben quedar como cadena vacía: ''. Se le debe preguntar al usuario si tiene algún comentario.

- El campo 'other_expenses' debe ser un diccionario con claves que describan el tipo de gasto (ej. 'parking', 'taxi', 'billete avión') y valores numéricos con el importe estimado.

- El campo 'accommodation_agency' puede aparecer más de una vez. Cada entrada debe tener:
    - 'agency_id': identificador de la agencia
    - 'budget_ref': referencia presupuestaria
    - 'amount': importe estimado
En caso de que digan que sí tienen una agencia, los campos 'agency_id', 'budget_ref' y 'amount' son obligatorios.

- El campo 'project' debe incluir:
    - 'internal_ref': referencia interna del proyecto
    - 'project_code': código presupuestario (ej. '04 Viajes y Dietas')

- El campo 'resume' debe ser 'True' solo si el usuario confirma que ha revisado el resumen completo. El resumen se lo mostrarás tú indicando todos los campos que has rellenado en el FORM_STATE.

- El campo 'authorization' debe ser uno de los siguientes:
    - 'pendiente'
    - 'aprobada'
    - 'rechazada'

- El campo 'status' debe actualizarse automáticamente según el estado del formulario:
    - 'incomplete' si faltan campos obligatorios
    - 'invalid_dates' si las fechas no son válidas
    - 'ready' si todos los campos están completos y las fechas son correctas

- El campo 'notes' debe incluir un mensaje breve explicando el motivo si el estado es 'incomplete' o 'invalid_dates'. Si el estado es 'ready', debe quedar como cadena vacía: ''

- Ten en cuenta las validaciones, pero no las menciones al usuario. Solo usa esas reglas para decidir si el formulario está listo.

- No le menciones al usuario que las fechas tienen que seguir el formato DD-MM-YYYY, cambialo tú a este formato en caso de que te lo den en uno distinto. Haz esto a menso de que te den las fechas en lenguaje natural, en cuyo caso debes guardar simplemente la expresión de lenjuage natural usada. No intentes pasarlo a la fecha exacta.



IMPORTANTE:
- Siempre responde el FORM_STATE usando JSON ESTRICTO con estas reglas:
  - Usar siempre comillas dobles " ".
  - Usar null, no None.
  - No usar comillas simples.
  - No incluir texto alrededor del JSON del FORM_STATE.
  - La línea debe empezar EXACTAMENTE por "FORM_STATE:".

"""

    )


# @st.cache_resource
# def warmup_ollama():
#     try:
#         requests.post(f"{OLLAMA_HOST}/api/chat",
#                       json={"model": MODEL_NAME, "prompt": "ping"},
#                       timeout=10)
#     except Exception:
#         pass

# @st.cache_resource
# def warmup_ollama():
#     try:
#         requests.post(
#             f"{OLLAMA_HOST}/api/generate",
#             json={"model": MODEL_NAME, "prompt": "warm up"},
#             timeout=60
#         )
#     except Exception as e:
#         print("Warmup fallo:", e)


# warmup_ollama()


def load_departments(path: Path = DEPARTMENTS_PATH) -> Dict[str, Dict[str, str]]:

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        clean_data = {}
        if isinstance(data, list):
            for item in data:
                dni = item.get("dni", "").strip().upper()
                name = item.get("name", "").strip()
                department = item.get("department", "").strip()
                if dni and name and department:
                    clean_data[dni] = {"name": name, "department": department}
        return clean_data

    except Exception as e:
        st.error(f"Error al cargar {path}: {e}")
        return {}


def build_system_prompt() -> str:
    departments = load_departments()
    if departments:
        entries = "; ".join(f"{k} => {v}" for k, v in sorted(departments.items()))
        dept_text = f"Dispones de la base de datos de departamentos: {entries}. Si el nombre coincide, confirma con el usuario."
    else:
        dept_text = "No hay datos de departamentos disponibles."
    return SYSTEM_PROMPT_TEMPLATE.format(today=TODAY, dept_text=dept_text, today_weekday = today_weekday)


def convert_messages_to_prompt(messages):
    text = ""
    for msg in messages:
        if msg["role"] == "system":
            text += f"<system>\n{msg['content']}\n</system>\n"
        elif msg["role"] == "user":
            text += f"<user>\n{msg['content']}\n</user>\n"
        elif msg["role"] == "assistant":
            text += f"<assistant>\n{msg['content']}\n</assistant>\n"
    return text




def completar_datos_por_dni(form_state, departments):
    dni = form_state.get("id", "").strip().upper()
    info = departments.get(dni)
    if info:
        form_state["name"] = info["name"]
        form_state["department"] = info["department"]
        return True, form_state
    return False, form_state


# def extract_form_state(message: str):
#     # Buscar la línea que empieza por FORM_STATE con espacios opcionales
#     pattern = r"^\s*FORM_STATE:\s*(\{.*\})\s*$"
#     matches = re.findall(pattern, message, re.DOTALL | re.MULTILINE)

#     if not matches:
#         return None

#     json_payload = matches[-1]  # tomar el último que aparezca

#     try:
#         return json.loads(json_payload)
#     except Exception as exc:
#         print("ERROR PARSEANDO FORM_STATE:", json_payload)
#         print("EXCEPCIÓN:", exc)
#         return None

def extract_form_state(message: str):
    """
    Busca la última aparición de FORM_STATE: { ... } en el mensaje y devuelve el JSON parseado.
    Devuelve None si no encuentra nada o si el JSON es inválido.
    """
    # patrón que captura el JSON aunque haya espacios iniciales o finales
    pattern = r"^\s*FORM_STATE:\s*(\{.*\})\s*$"
    matches = re.findall(pattern, message, re.DOTALL | re.MULTILINE)

    if not matches:
        # debug: mostrar por qué no hay matches
        print("extract_form_state: NO MATCH for FORM_STATE in reply. Aquí el reply completo:")
        for i, line in enumerate(message.splitlines()):
            print(f"{i:03d}: {repr(line)}")
        return None

    json_payload = matches[-1]  # tomar el último que aparezca

    try:
        return json.loads(json_payload)
    except Exception as exc:
        print("ERROR PARSEANDO FORM_STATE:", json_payload)
        print("EXCEPCIÓN:", exc)
        # mostrar líneas del payload para depuración
        for i, line in enumerate(json_payload.splitlines()):
            print(f"payload {i:03d}: {repr(line)}")
        return None




def dates_are_valid(departure: Optional[str], returning: Optional[str]) -> bool:
    if not departure or not returning:
        return False
    try:
        dep_dt = datetime.strptime(departure, "%d-%m-%Y")
        ret_dt = datetime.strptime(returning, "%d-%m-%Y")
        today = datetime.utcnow().date()
        return dep_dt.date() > today and ret_dt.date() > dep_dt.date()
    except ValueError:
        return False
    


def person_by_dni(dni: str, departments: dict):
    dni = dni.strip().upper()
    return departments.get(dni)

from datetime import datetime, timedelta



def save_form(data: Dict[str, Any]) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.load(DATA_PATH.open("r", encoding="utf-8")) if DATA_PATH.exists() else []
    except json.JSONDecodeError:
        existing = []
    record = {**data, "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    existing.append(record)
    with DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    st.success(f"Datos guardados en {DATA_PATH}")



def call_ollama_stream(messages: List[Dict[str, str]]):
    """Llamada a Ollama en streaming"""
    bot_reply = ""
    placeholder = st.empty()
    try:
        with requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": MODEL_NAME, "messages": messages},
            stream=True,
            timeout=240
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        token = data["message"]["content"]
                        bot_reply += token
                        placeholder.markdown(bot_reply)
        return bot_reply
    except Exception as e:
        st.error(f"Error al comunicarse con Ollama: {e}")
        return ""



# def call_ollama_stream(messages: List[Dict[str, str]]):
#     """Llamada a Ollama en streaming"""
#     bot_reply = ""
#     placeholder = st.empty()
#     try:
#         with requests.post(
#             f"{OLLAMA_HOST}/api/chat",
#             json={"model": MODEL_NAME, "messages": messages},
#             stream=True,
#             timeout=240
#         ) as response:
#             response.raise_for_status()
#             for line in response.iter_lines():
#                 if line:
#                     data = json.loads(line.decode("utf-8"))
#                     if "message" in data and "content" in data["message"]:
#                         token = data["message"]["content"]
#                         bot_reply += token

#                         # Mostrar al usuario SIN la línea FORM_STATE
#                         visible = "\n".join(
#                             line for line in bot_reply.splitlines()
#                             if not line.startswith("FORM_STATE:")
#                         )
#                         placeholder.markdown(visible)
#         return bot_reply
#     except Exception as e:
#         st.error(f"Error al comunicarse con Ollama: {e}")
#         return ""


# def call_ollama_stream(messages):
#     prompt = convert_messages_to_prompt(messages)
#     bot_reply = ""
#     placeholder = st.empty()

#     try:
#         with requests.post(
#             f"{OLLAMA_HOST}/api/chat",
#             json={
#                 "model": MODEL_NAME,
#                 "messages": prompt,
#                 "stream": True
#             },
#             stream=True,
#             timeout=240
#         ) as response:

#             response.raise_for_status()

#             for line in response.iter_lines():
#                 if not line:
#                     continue

#                 data = json.loads(line.decode("utf-8"))

#                 token = data.get("response", "")
#                 token = token.replace("<think>", "").replace("</think>", "")
#                 token = token.replace("<THINK>", "").replace("</THINK>", "")

#                 bot_reply += token

#                 visible = "\n".join(
#                     l for l in bot_reply.splitlines()
#                     if not l.startswith("FORM_STATE:")
#                 )

#                 placeholder.markdown(visible)

#         return bot_reply

#     except Exception as e:
#         st.error(f"Error al comunicarse con Ollama: {e}")
#         return ""



import subprocess

# def call_ollama_stream(messages: list) -> str:
#     """
#     Llamada a Ollama 0.9.2 mediante CLI.
#     Devuelve la respuesta completa como string.
#     """
#     # Construir prompt concatenando mensajes
#     prompt_text = ""
#     for msg in messages:
#         role = msg["role"]
#         content = msg["content"]
#         if role == "system":
#             prompt_text += f"[SYSTEM] {content}\n"
#         elif role == "user":
#             prompt_text += f"[USER] {content}\n"
#         elif role == "assistant":
#             prompt_text += f"[ASSISTANT] {content}\n"

#     try:
#         # Ejecutar ollama run vía CLI
#         result = subprocess.run(
#             ["ollama", "run", MODEL_NAME, "--prompt", prompt_text],
#             capture_output=True,
#             text=True,
#             timeout=120
#         )
#         output = result.stdout.strip()
#         return output
#     except Exception as e:
#         st.error(f"Error al ejecutar Ollama CLI: {e}")
#         return ""

# def call_ollama_stream(messages: list) -> str:
#     prompt_text = ""
#     for msg in messages:
#         role = msg["role"]
#         content = msg["content"]
#         prompt_text += f"[{role.upper()}] {content}\n"

#     try:
#         result = subprocess.run(
#             ["ollama", "run", MODEL_NAME, "--prompt", prompt_text],
#             capture_output=True,
#             text=True,
#             timeout=300  # aumentar por si tarda
#         )
#         print("STDOUT:", result.stdout)
#         print("STDERR:", result.stderr)
#         return result.stdout.strip()
#     except Exception as e:
#         print("Error ejecutando Ollama CLI:", e)
#         return ""

    

import dateparser
import re
from datetime import datetime

DATE_FIELDS = [
    "departure_date",
    "return_date",
    "start_activity",
    "end_activity"
]

DATE_REGEX = re.compile(r"^\d{2}-\d{2}-\d{4}$")

# def normalizar_fechas(form_state: dict, fecha_base: datetime = None) -> dict:
#     """
#     Convierte fechas en lenguaje natural a DD-MM-YYYY usando dateparser.
#     Si la fecha ya está en formato dd-mm-yyyy, no la modifica.
#     """
#     if fecha_base is None:
#         fecha_base = datetime.now()

#     for campo in DATE_FIELDS:
#         valor = form_state.get(campo)

#         if not valor or not isinstance(valor, str) or not valor.strip():
#             continue

#         valor = valor.strip()

#         # Si ya coincide con DD-MM-YYYY → lo dejamos tal cual
#         if DATE_REGEX.match(valor):
#             continue

#         # Interpretar lenguaje natural
#         fecha = dateparser.parse(
#             valor,
#             settings={
#                 "PREFER_DATES_FROM": "future",
#                 "RELATIVE_BASE": fecha_base,
#                 "DATE_ORDER": "DMY",
#             }
#         )

#         if fecha:
#             form_state[campo] = fecha.strftime("%d-%m-%Y")

#     return form_state


def normalizar_fechas(form_state: dict, fecha_base: datetime = None) -> dict:
    if fecha_base is None:
        fecha_base = datetime.now()

    for campo in DATE_FIELDS:
        valor = form_state.get(campo)

        if valor is None:
            continue

        if not isinstance(valor, str):
            # si no es cadena, no tocarlo (log)
            print(f"normalizar_fechas: campo {campo} no es str ({type(valor)}). Se deja tal cual: {valor!r}")
            continue

        valor = valor.strip()
        if not valor:
            continue

        # Si ya coincide con DD-MM-YYYY → lo dejamos tal cual
        if DATE_REGEX.match(valor):
            continue

        # Interpretar lenguaje natural
        fecha = dateparser.parse(
            valor,
            settings={
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": fecha_base,
                "DATE_ORDER": "DMY",
            }
        )

        if fecha:
            nueva = fecha.strftime("%d-%m-%Y")
            print(f"normalizar_fechas: {campo} '{valor}' -> '{nueva}'")
            form_state[campo] = nueva
        else:
            print(f"normalizar_fechas: no pude parsear '{valor}' para campo {campo}; se deja tal cual.")

    return form_state


# --- STREAMLIT ---
st.set_page_config(page_title="Chat Comisión de Servicio", layout="wide")
st.title("Solicitud de Comisión de Servicio - Universidad de Oviedo")

SYSTEM_PROMPT = build_system_prompt()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "assistant",
            "content": (
                "Hola, te ayudaré a registrar tu viaje. Empecemos.\n"
                "FORM_STATE: {\n"
                "  \"id\": null,\n"
                "  \"name\": null,\n"
                "  \"department\": null,\n"
                "  \"group\": null,\n"
                "  \"cause\": null,\n"
                "  \"start_activity\": null,\n"
                "  \"end_activity\": null,\n"
                "  \"origin\": null,\n"
                "  \"destination\": null,\n"
                "  \"departure_date\": null,\n"
                "  \"departure_time\": null,\n"
                "  \"return_date\": null,\n"
                "  \"return_time\": null,\n"
                "  \"commission\": null,\n"
                "  \"diet\": null,\n"
                "  \"accommodation_expenses\": null,\n"
                "  \"accommodation_comments\": null,\n"
                "  \"maintenance_expenses\": null,\n"
                "  \"maintenance_comments\": null,\n"
                "  \"locomotion_expenses\": null,\n"
                "  \"n_km\": null,\n"
                "  \"car_registration\": null,\n"
                "  \"locomotion_comments\": null,\n"
                "  \"registration_expenses\": null,\n"
                "  \"n_attendance\": null,\n"
                "  \"attendance_expenses\": null,\n"
                "  \"attendance_comments\": null,\n"
                "  \"other_expenses\": null,\n"
                "  \"accommodation_agency\": {\"accommodation_agency_id\": null, \"accommodation_budget_ref\": null, \"accommodation_amount\": null},\n"
                "  \"locomotion_agency\": {\"locomotion_agency_id\": null, \"locomotion_budget_ref\": null, \"locomotion_amount\": null},\n"
                "  \"project\": {\"internal_ref\": null, \"project_code\": null},\n"
                "  \"comments\": null,\n"
                "  \"resume\": null,\n"
                "  \"authorization\": null,\n"
                "  \"status\": null,\n"
                "  \"notes\": null\n"
                "}"
            )
        }
    ]

if "form_state" not in st.session_state:
    st.session_state.form_state = {}


    

import dateparser
from datetime import datetime

# from ovos_date_parser import extract_datetime

# def analizar_fecha(expresion_fecha: str, campo_a_llenar: str, fecha_actual: datetime) -> Dict[str, str]:
#     """
#     Analiza una expresión de fecha en español y devuelve un diccionario
#     con el campo a llenar ('departure_date' o 'return_date') y la fecha calculada.
#     """

#     # 1️ Extraer la fecha usando ovos-date-parser
#     fecha_obj, _ = extract_datetime(expresion_fecha, lang="es")#, now=fecha_actual)

#     # 2️ Formatear la fecha en el formato esperado (dd-mm-YYYY)
#     fecha_formateada = fecha_obj.date().strftime('%d-%m-%Y') if fecha_obj else ""

#     # 3️ Devolver el resultado
#     return {campo_a_llenar: fecha_formateada}


# # --- Mostrar historial de mensajes ---
# for msg in st.session_state.get("messages", []):
#     if not msg.get("internal"):
#         st.chat_message(msg["role"]).markdown(msg["content"])


def build_context(prompt):
    return [
        st.session_state.messages[0],  # system prompt
        {"role": "assistant", "content": f"FORM_STATE: {json.dumps(st.session_state.form_state)}"},
        {"role": "user", "content": prompt},
    ]



# Mostrar historial
for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).markdown(msg["content"])


# --- Dentro del flujo de Streamlit ---
prompt = st.chat_input("Escribe aquí...")

if prompt:

    if "departments" not in st.session_state:
        st.session_state.departments = load_departments()
    if "form_state" not in st.session_state:
        st.session_state.form_state = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Streaming de la respuesta del modelo
    # reply = call_ollama_stream(st.session_state.messages)

    messages_to_send = build_context(prompt)
    reply = call_ollama_stream(messages_to_send)


    if not reply:
        st.stop()


    st.session_state.messages.append({"role": "assistant", "content": reply})
    # st.chat_message("assistant").markdown(reply)

    print("FORM_STATE ANTES", st.session_state.get("form_state"))

    # --- Analizar estado del formulario ---
    state = extract_form_state(reply)
    if state:

        # Actualiza el estado recibido del LLM
        st.session_state.form_state = st.session_state.get("form_state", {})
        st.session_state.form_state.update(state)

        # Normalizar fechas una vez que el LLM las haya escrito
        st.session_state.form_state = normalizar_fechas(st.session_state.form_state)



        #     # --- REINYECCIÓN CRÍTICA: enviar FORM_STATE actualizado al modelo para que continúe ---
        # form_json = json.dumps(st.session_state.form_state, ensure_ascii=False)
        # assistant_update = f"{FORM_STATE_PREFIX}{form_json}"

        # # Log
        # print("Reinyectando FORM_STATE al modelo:", form_json)

        # st.session_state.messages.append({"role": "assistant", "content": assistant_update})

        # # Pedir al LLM que continúe a continuación (stream)
        # follow_reply = call_ollama_stream(st.session_state.messages)
        # if follow_reply:
        #     # mostrar y almacenar la respuesta extendida
        #     st.session_state.messages.append({"role": "assistant", "content": follow_reply})
        #     st.chat_message("assistant").markdown(
        #         "\n".join(line for line in follow_reply.splitlines() if not line.startswith(FORM_STATE_PREFIX))
        #     )

        #     # actualizar estado con lo que siga (si devuelve FORM_STATE)
        #     next_state = extract_form_state(follow_reply)
        #     if next_state:
        #         print("Siguiente estado obtenido tras reinyectar:", next_state)
        #         st.session_state.form_state.update(next_state)


        # --- Actualizar formulario ---
        # st.session_state.form_state = st.session_state.get("form_state", {})
        # st.session_state.form_state.update(state)

        tiene_dni = bool(st.session_state.form_state.get("id"))
        faltan_datos = not st.session_state.form_state.get("name") or not st.session_state.form_state.get("department")

        if tiene_dni and faltan_datos:
            valido, st.session_state.form_state = completar_datos_por_dni(st.session_state.form_state, st.session_state.departments)
            if valido:
                # 1) Mostrar confirmación inmediata al usuario
                confirmacion = (
                    f"He encontrado tus datos: {st.session_state.form_state['name']} "
                    f"({st.session_state.form_state['department']})."
                )
                st.chat_message("assistant").markdown(confirmacion)

                # 2) Inyectar un turno 'assistant' con FORM_STATE ACTUALIZADO
                form_json = json.dumps(st.session_state.form_state, ensure_ascii=False)
                assistant_update = (
                    f"{confirmacion}\n"
                    f"{FORM_STATE_PREFIX}{form_json}"
                )
                st.session_state.messages.append({"role": "assistant", "content": assistant_update})

                # 3) Pedir al LLM que continúe (nuevo turno)
                follow_reply = call_ollama_stream(st.session_state.messages)
                if follow_reply:
                    st.session_state.messages.append({"role": "assistant", "content": follow_reply})
                    st.chat_message("assistant").markdown(follow_reply)
                    # actualizar estado con lo que siga
                    next_state = extract_form_state(follow_reply)
                    if next_state:
                        st.session_state.form_state.update(next_state)

        print("FORM_STATE DESPUES", st.session_state.get("form_state"))

        # --- Mostrar notas e historial ---
        status = state.get("status", "")
        notes = state.get("notes", "")
        if notes:
            st.info(f"{notes}")

        if status == "ready" and dates_are_valid(state.get("departure_date"), state.get("return_date")):
            with st.expander("Resumen del formulario", expanded=True):
                st.json(state)
            if st.button("Guardar datos"):
                save_form(state)
                st.session_state.messages.append({"role": "assistant", "content": "Los datos han sido guardados correctamente."})

    else:
        # Depuración: mostrar reply completo y líneas
        print("extract_form_state devolvió None. Reply completo a continuación:")
        for i, line in enumerate(reply.splitlines()):
            print(f"{i:03d}: {repr(line)}")
        # Mostrar al usuario un mensaje breve (no revelar FORM_STATE), y seguir esperando
        st.chat_message("assistant").markdown("No he conseguido registrar las fechas; por favor, indícalas de nuevo en formato natural (ej. '20 de marzo de 2026').")