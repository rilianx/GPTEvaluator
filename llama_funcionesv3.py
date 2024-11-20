import os 
import sys
import pandas as pd
import ollama
import json
from zipfile import ZipFile
import re 
from datetime import datetime, timedelta
from framework_experimentos import (
    convert_gpt_scores, optimize_params, load_dataset, generate_prompts, calculate_mse, save_stats, read_results, generate_sets
)

from framework_experimentos import *


"""

888~-_                                             d8          
888   \  888-~\  e88~-_  888-~88e-~88e 888-~88e  _d88__  d88~\ 
888    | 888    d888   i 888  888  888 888  888b  888   C888   
888   /  888    8888   | 888  888  888 888  8888  888    Y88b  
888_-~   888    Y888   ' 888  888  888 888  888P  888     888D 
888      888     "88_-~  888  888  888 888-_88"   "88_/ \_88P  
                                       888                     
                                       
"""


def build_prompt_mistral(user_message, system_message=None):
    """
    Construye el prompt para el modelo Mistral:latest.
    """
    if system_message:
        prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST]"
    else:
        prompt = f"<s>[INST] {user_message} [/INST]"
    return prompt

def build_prompt_mistral_nemo(user_message, system_message=None):
    """
    Construye el prompt para el modelo Mistral-Nemo:latest.
    """
    if system_message:
        prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST]"
    else:
        prompt = f"<s>[INST] {user_message} [/INST]"
    return prompt

def build_prompt_gemma2_2b(user_message, system_message=None):
    """
    Construye el prompt para el modelo Gemma2:2b.
    """
    prompt = ""
    if system_message:
        prompt += f"<start_of_turn>system\n{system_message}<end_of_turn>\n"
    prompt += f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
    return prompt

def build_prompt_gemma2_latest(user_message, system_message=None):
    """
    Construye el prompt para el modelo Gemma2:latest.
    """
    prompt = ""
    if system_message:
        prompt += f"<start_of_turn>system\n{system_message}<end_of_turn>\n"
    prompt += f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
    return prompt

def build_prompt_llama(user_message, system_message=None):
    """
    Construye el prompt para el modelo Llama 3.1 y 3.2.
    """
    if system_message:
        prompt = f"system\n{system_message}\n"
    else:
        prompt = "\n"
    prompt += f"user\n{user_message}\nPlease provide your response in the requested format: {{'score': your_score}}."
    return prompt



def configure_prompt(model_name, user_message, system_message=None):
    """
    Configura el formato del prompt utilizando las funciones específicas de construcción de prompts
    según el modelo seleccionado.
    """

    if "mistral:latest" in model_name:
        prompt = build_prompt_mistral(user_message, system_message)
    elif "mistral-nemo:latest" in model_name:
        prompt = build_prompt_mistral_nemo(user_message, system_message)
    elif "gemma2:2b" in model_name:
        prompt = build_prompt_gemma2_2b(user_message, system_message)
    elif "gemma2:latest" in model_name:
        prompt = build_prompt_gemma2_latest(user_message, system_message)
    elif "llama" in model_name:
        prompt = build_prompt_llama(user_message, system_message)
    else:
        raise ValueError(f"El modelo '{model_name}' no está configurado para su uso.")
    
    return prompt

"""

888~~\                             
888   |   /~~~8e   d88~\  e88~~8e  
888 _/        88b C888   d888  88b 
888  \   e88~-888  Y88b  8888__888 
888   | C888  888   888D Y888    , 
888__/   "88_-888 \_88P   "88___/  
                                   
                            
"""


# Retorna un dataset con el MSE por grupo
def calculate_mse(result_set, normalize):

    # Eliminar filas con NaN en 'real_eval' o 'gpt_eval'
    result_set = result_set.dropna(subset=['real_eval', 'gpt_eval'])


    if normalize:
        mse_dict = result_set.groupby('dataset')[result_set.columns.tolist()].apply(lambda x: mean_squared_error(x['real_eval']/3, x['gpt_eval']/3)).to_dict()
        overall_mse = mean_squared_error(result_set['real_eval']/3, result_set['gpt_eval']/3)
    else:
        mse_dict = result_set.groupby('dataset')[result_set.columns.tolist()].apply(lambda x: mean_squared_error(x['real_eval'], x['gpt_eval'])).to_dict()
        overall_mse = mean_squared_error(result_set['real_eval'], result_set['gpt_eval'])

    mse_dict['All'] = overall_mse
    return mse_dict

# Obtiene los parámetros óptimos para disminuir el error
def optimize_params(gpt_dicts, real_scores, criteria, eval_function):
    criteria_scores = get_x(gpt_dicts, criteria)

    # Verificar la longitud de los datos
    if len(criteria_scores) != len(real_scores):
        print(f"Longitudes inconsistenes: criteria_scores ({len(criteria_scores)}) y real_scores ({len(real_scores)})")
        # Recortar ambas listas a la longitud mínima común para evitar errores de transmisión
        min_length = min(len(criteria_scores), len(real_scores))
        criteria_scores = criteria_scores[:min_length]
        real_scores = real_scores[:min_length]

    if eval_function == "map":
        optimizer = MapOptimizer(criteria_scores, real_scores)
    elif eval_function == "cuts":
        optimizer = CutsOptimizer(criteria_scores, real_scores)
    else:
        raise ValueError(f"El valor de 'eval_function' no es válido: {eval_function}. Debe ser 'map' o 'cuts'.")

    return optimizer.params


# Función para construir el prompt en formato correcto para Llama 3.1
def build_prompt_llama(user_message, system_message=None):
    if system_message:
        prompt = f"system\n{system_message}\n"
    else:
        prompt = "\n"
    prompt += f"user\n{user_message}\nPlease provide your response in the requested format: {{'score': your_score}}. This format follows the structure of a Python dictionary , using key-value pairs for clarity and data organization. Additional criteria may be included as needed.\n"
    return prompt

# Procesa los textos uno a uno utilizando llama 3.1
def llama_multiple_llama(texts, model="llama3.1", system_message=None):
    answers = [None] * len(texts)
    for idx, text in enumerate(texts):
        prompt = build_prompt_llama(text, system_message)
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        answers[idx] = [response['message']['content']]
    return answers

# Función para evaluar el modelo con un DataFrame y un prompt
def eval_llama(df, prompt_template, model="llama3.1", system_message=None):
    texts = []
    for _, row in df.iterrows():
        text = prompt_template.format(Question=row['question'], Answer=row['answer'], Context=row['context'])
        texts.append(text)
    
    answers_llama = llama_multiple_llama(texts, model=model, system_message=system_message)
    return answers_llama

# Extrae diccionarios de las respuestas generadas por llama 3.1
def extract_dicts_llama(answers_llama):
    pattern = r'\{[^{}]+\}'
    llama_dicts = []
    for answer_llama in answers_llama:
        try:
            # Buscar el diccionario en formato de texto
            answer = re.findall(pattern, answer_llama[0])[0]
            parsed_dict = eval(answer)  # Convertir el texto en un diccionario

            # Verificar que 'score' esté presente en el diccionario
            if 'score' in parsed_dict:
                llama_dicts.append(parsed_dict)
            else:
                print("Advertencia: Respuesta sin campo 'score' encontrada.")
                llama_dicts.append(None)  # Agregar None si el 'score' no está presente

        except Exception as e:
            print(f"Error al extraer diccionario. Respuesta llama 3.1: \n{answer_llama[0]}\n\n")
            llama_dicts.append(None)  # Agregar None para respuestas no válidas
    
    return llama_dicts


# Limpia las filas del dataset donde hubo errores en las respuestas
def clean_set_llama(dataset, llama_dicts, criteria):
    for i in reversed(range(len(llama_dicts))):
        if llama_dicts[i] is None or 'score' not in llama_dicts[i]:
            llama_dicts.pop(i)
            if i in dataset.index:
                dataset.drop(index=i, inplace=True)
        elif not all(key in llama_dicts[i] for key in criteria):
            llama_dicts.pop(i)
            if i in dataset.index:
                dataset.drop(index=i, inplace=True)

"""

  d8                    ,e,                
_d88__ 888-~\   /~~~8e   "  888-~88e       
 888   888          88b 888 888  888       
 888   888     e88~-888 888 888  888       
 888   888    C888  888 888 888  888       
 "88_/ 888     "88_-888 888 888  888       
                                           
"""
# Entrena el modelo llama 3.1
def train_llama(train_set, prompt_template, criteria, eval_function, model, temperature, system_message=None):
    train_set = train_set.copy()
    answers_llama = eval_llama(train_set, prompt_template, model, system_message=system_message)
    llama_dicts = extract_dicts_llama(answers_llama)
    clean_set_llama(train_set, llama_dicts, criteria)
    real_scores = train_set['real_eval'].tolist()
    return optimize_params(llama_dicts, real_scores, criteria, eval_function)
"""

  d8                      d8   
_d88__  e88~~8e   d88~\ _d88__ 
 888   d888  88b C888    888   
 888   8888__888  Y88b   888   
 888   Y888    ,   888D  888   
 "88_/  "88___/  \_88P   "88_/ 
                               
                               
"""
# Calcula el MSE del set de prueba
def test_llama(test_set, prompt_template, criteria, eval_function, eval_params, model, temperature, system_message=None, normalize_mse=False):
    """
    Calcula el MSE del set de prueba utilizando el modelo LLaMA.
    """
    test_set = test_set.copy()
    answers_llama = eval_llama(test_set, prompt_template, model, system_message=system_message)
    llama_dicts = extract_dicts_llama(answers_llama)

    # Limpiar el test_set y llama_dicts para asegurar que tienen la misma longitud
    clean_set_llama(test_set, llama_dicts, criteria)
    test_set.reset_index(drop=True, inplace=True)
    
    # Filtrar `llama_dicts` para incluir solo los elementos que tienen `score`
    filtered_llama_dicts = [entry for entry in llama_dicts if entry is not None and 'score' in entry]
    
    # Obtener los puntajes reales después de limpiar
    real_scores = test_set['real_eval'].tolist()
    
    # Generar puntajes predichos
    pred_scores = convert_gpt_scores(filtered_llama_dicts, real_scores, criteria, eval_function, eval_params)

    # Verificar la longitud de pred_scores y real_scores después de la limpieza
    if len(pred_scores) != len(real_scores):
        print("Error: Las longitudes de pred_scores y real_scores no coinciden después de filtrar respuestas.")
        print(f"Longitud de pred_scores: {len(pred_scores)}, Longitud de real_scores: {len(real_scores)}")
        raise ValueError(f"Las longitudes de pred_scores ({len(pred_scores)}) y real_scores ({len(real_scores)}) no coinciden.")

    result_set = test_set.copy(deep=True)
    result_set['gpt_eval'] = pred_scores

    df_dicts = pd.DataFrame(filtered_llama_dicts)
    result_set = pd.concat([result_set, df_dicts], axis=1)

    return calculate_mse(result_set, normalize_mse), result_set

"""

                                              ,e,                                    d8   
 e88~~8e  Y88b  /  888-~88e   e88~~8e  888-~\  "  888-~88e-~88e  e88~~8e  888-~88e _d88__ 
d888  88b  Y88b/   888  888b d888  88b 888    888 888  888  888 d888  88b 888  888  888   
8888__888   Y88b   888  8888 8888__888 888    888 888  888  888 8888__888 888  888  888   
Y888    ,   /Y88b  888  888P Y888    , 888    888 888  888  888 Y888    , 888  888  888   
 "88___/   /  Y88b 888-_88"   "88___/  888    888 888  888  888  "88___/  888  888  "88_/ 
                   888                                                                    
                   
"""

# Función principal para ejecutar el experimento con Llama 3.1
def experiment_llama(dataset, prompts, repetitions, eval_function, eval_params=None, train_set_size=40, test_set_size=60, seed=42, model="llama3.1", temperature=0.1,normalize_mse=False,repeat_test_set=True):
    sets = generate_sets(dataset, repetitions, train_set_size, test_set_size, seed,repeat_test_set)
    stats = []
    full_df = pd.DataFrame()

    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data.prompt
        criteria = prompt_data.criteria
        stats.append([])

        for j in range(repetitions):
            train_set = sets[j].train_set
            test_set = sets[j].test_set

            iter_params = eval_params
            if not eval_params:
                print(f"Entrenando Prompt {i+1} con Train Set {j+1}")
                iter_params = train_llama(train_set, prompt, criteria, eval_function, model, temperature)
                print()

            print(f"Evaluando Prompt {i+1} con Test Set {j+1}")
            metrics, result_set = test_llama(test_set, prompt, criteria, eval_function, iter_params, model, temperature, normalize_mse)
            stats[i].append({
                "MSE": metrics,
                "params": iter_params
            })
            print()

            result_set['prompt'] = prompt
            result_set['repetition'] = j+1
            full_df = pd.concat([full_df, result_set], ignore_index=True)

    full_df.to_excel(f"{filename}.xlsx", index=False)
    filename = save_stats(prompts, stats, eval_function, train_set_size, test_set_size, model, temperature)
    read_results(filename)

    return stats

def sanitize_sheet_name(sheet_name):
    return sheet_name.replace(":", "_")

def generate_formatted_filename(models, extension="zip"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    models_str = "_".join(models)
    filename = f"{timestamp}_{models_str}.{extension}"
    return filename



"""
  d8                      d8                                      888           888        
_d88__  e88~~8e   d88~\ _d88__       888-~88e-~88e  e88~-_   e88~\888  e88~~8e  888  d88~\ 
 888   d888  88b C888    888         888  888  888 d888   i d888  888 d888  88b 888 C888   
 888   8888__888  Y88b   888         888  888  888 8888   | 8888  888 8888__888 888  Y88b  
 888   Y888    ,   888D  888         888  888  888 Y888   ' Y888  888 Y888    , 888   888D 
 "88_/  "88___/  \_88P   "88_/       888  888  888  "88_-~   "88_/888  "88___/  888 \_88P  
                                                                                                                                                                                                                                      
"""

def test_multiple_models(models_to_test, df, prompts, repetitions, eval_function="map", train_set_size=40, 
                         test_set_size=60, seed=42, temperature=0.1, normalize_mse=False):
    """
    Prueba una lista de modelos especificados en 'models_to_test' en un experimento.
    """
    model_results = {}
    
    # Generar nombre de archivo Excel con formato
    excel_filename = generate_formatted_filename(models_to_test, extension="xlsx")

    # Crear un archivo Excel con una hoja para cada modelo
    with pd.ExcelWriter(excel_filename) as writer:
        for model_name in models_to_test:
            print(f"\nProbando el modelo: {model_name}")

            # Generar prompt específico para el modelo
            user_message = "Tu mensaje principal aquí"  # Aquí podrías personalizar el mensaje del usuario
            system_message = "Instrucciones del sistema aquí"  # Mensaje del sistema si es necesario
            prompt_template = configure_prompt(model_name, user_message, system_message)

            # Ejecutar el experimento con el prompt configurado
            metrics = experiment_llama(df, prompts, repetitions, eval_function, train_set_size=train_set_size,
                                       test_set_size=test_set_size, seed=seed, model=model_name, 
                                       temperature=temperature, normalize_mse=normalize_mse)
            
            # Reemplazar caracteres no permitidos en el nombre de la hoja
            sanitized_sheet_name = sanitize_sheet_name(model_name)
            
            # Guardar los resultados en una hoja con el nombre del modelo
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_excel(writer, sheet_name=sanitized_sheet_name)
            
            # Guardar los resultados para el modelo actual en el diccionario
            model_results[model_name] = metrics
    
    # Generar nombre de archivo ZIP con formato
    zip_filename = generate_formatted_filename(models_to_test, extension="zip")
    with ZipFile(zip_filename, 'w') as zipf:
        zipf.write(excel_filename)

    print(f"\nArchivo Excel guardado como '{excel_filename}' y comprimido en '{zip_filename}'.")

    return model_results





"""
                        ,e,          
888-~88e-~88e   /~~~8e   "  888-~88e 
888  888  888       88b 888 888  888 
888  888  888  e88~-888 888 888  888 
888  888  888 C888  888 888 888  888 
888  888  888  "88_-888 888 888  888 
                                     
"""

def main():
    """
    Función principal para ejecutar el experimento en modelos específicos.
    """
    # Define manualmente los modelos a probar
    models_to_test = ["mistral:latest", "gemma2:2b", "gemma2:latest", "llama3.2:latest"]
    
    # Cargar el dataset
    column_data = {
        "context": "Contexto simple",
        "question": "Pregunta",
        "answer": "Respuesta",
        "real_eval": "Promedio Redondeado",
        "dataset": "DataSet"
    }
    df = load_dataset("datasets_v2.xlsx", "AllDatasets (1dif)", column_data)

    # Generar prompts
    prompt_data = {
        "question": "question.txt",
        "answer": "answer.txt",
        "instructions": {
            "score": "score_single.txt"
        }
    }
    prompt_folder = "GPTEvaluator/Experiments/Miniprompts_v2"
    prompts = generate_prompts(prompt_data, prompt_folder)

    # Parámetros del experimento
    repetitions = 5
    train_set_size = 40
    test_set_size = 60
    eval_function = "map"
    normalize_mse = False

    # Ejecutar pruebas en múltiples modelos
    model_results = test_multiple_models(models_to_test, df, prompts, repetitions, eval_function, 
                                         train_set_size, test_set_size, seed=42, temperature=0.1, 
                                         normalize_mse=normalize_mse)

    # Mostrar resultados
    print("\nResultados de MSE por modelo:")
    for model, result in model_results.items():
        print(f"Modelo: {model}, Resultados: {result}")

if __name__ == "__main__":
    main()

