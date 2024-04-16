
import pandas as pd
import re

def load_correct_answers(file_path):
    """
    Carga y procesa la rúbrica desde un archivo Excel.
    Extrae etiquetas de las preguntas para todas las hojas del documento.
    """
    # Cargar todas las hojas del archivo Excel
    all_sheets = pd.read_excel(file_path, sheet_name=None)

    # Diccionario para almacenar DataFrames procesados
    processed_dfs = {}

    # Procesar cada hoja
    for sheet_name, df in all_sheets.items():
        processed_dfs[sheet_name] = df

    return pd.concat(processed_dfs, axis=0, ignore_index=True)


def load_student_answers(file_paths):
    """
    Carga los datos del estudiante desde un archivo Excel y procesa la información paralela.
    """
    df_melted = []

    for id_control, file_path in file_paths:
      df = pd.read_excel(file_path)
      question_columns = [col for col in df.columns if "Pregunta" in col or "Respuesta" in col]
      questions = [col for col in question_columns if "Pregunta" in col]
      answers = [col for col in question_columns if "Respuesta" in col]
      # Melt the dataframe

      for q, a in zip(questions, answers):
          temp_df = df[['Nombre', 'Apellido(s)', 'Dirección de correo', q, a]].copy()
          temp_df.columns = ['Nombre', 'Apellido(s)', 'Dirección de correo', 'Pregunta', 'Respuesta Estudiante']
          temp_df['id_pregunta'] = int(q.split(" ")[1])
          # Add id_control column
          temp_df['id_control'] = id_control
          df_melted.append(temp_df)

    df_melted = pd.concat(df_melted, axis=0, ignore_index=True)

    df_melted['fullname'] = df_melted.apply(lambda row: row['Nombre'] + ' ' + row['Apellido(s)'], axis=1)

    # Rearrange columns
    df_melted = df_melted[['fullname', 'Dirección de correo', 'id_control', 'id_pregunta', 'Pregunta', 'Respuesta Estudiante']]

    return df_melted


def chat_gpt_multiple(api, texts):
    texts = texts
    # Definir una función interna para hacer las peticiones
    def make_requests():
        for idx, text in enumerate(texts):
            api.request(data={
                "messages": [{
                    "role": "user",
                    "content": text
                }]
            }, metadata={'index': idx}, max_retries=3, retry_max=10, retry_multiplier=1)

    # Ejecutar la función de peticiones
    api.run_request_function(make_requests)

    # Procesar y recopilar las respuestas en orden

    answers = [None] * len(texts)
    for result in api:
        idx = result.metadata['index']
        print(idx, end="-")
        answers[idx]=[]
        for choice in result.response['choices']:
          response = choice['message']['content']
          answers[idx].append(response)
    return answers


def extract_tuple_score(gpt_answer):
    # String de ejemplo que contiene tuplas
    texto = gpt_answer

    # Utilizamos una expresión regular para buscar tuplas en el formato (x, y)
    patron = r'\((\d+,\s*\d+(?:,\s*\d+)*)\)'
    tuplas_encontradas = re.findall(patron, texto)

    # Obtenemos la primera tupla si hay al menos una encontrada
    if tuplas_encontradas:
      #return tuplas_encontradas[0]
      tup = [int(num) for num in tuplas_encontradas[0].replace(" ", "").strip("()").split(",")]
      return tup
    else:
      return None

def extract_feedback(text):
  indice = text.find("(2)")
  feedback = None

  # Verifica si la subcadena fue encontrada
  if indice != -1:
      # Extrae la porción de la cadena hasta (2), sin incluir (2)
      feedback = text[:indice]
  return feedback

def get_gpt_answers(dataset, template):
    ids = []; texts=[]
    for i, row in dataset.iterrows():
        # Extracting data from the dataset row
        text = template.format(Pregunta=row['Pregunta'], RespuestaA=row['Respuesta Estudiante'], RespuestaC=row['Respuesta'])
        texts.append(text)
        ids.append((row['fullname'],row['id_control'],row['id_pregunta']))

    gpt_answers = chat_gpt_multiple(texts)

    return ids, gpt_answers

def tuple2score(tup):
   score = round((0.6*tup[1]+0.4*tup[2]))
   return 3 if score >= 8 else (2 if score >= 6 else (1 if score >= 4 else 0))
   

def write_evaluations(dataset, ids, gpt_answers):
    for k in range(len(gpt_answers)):
      ans = []
      if gpt_answers[k] is None: continue

      for answer in gpt_answers[k]:
        try:
          # Extracting scores for correctness, clarity, and relevance
          tup = extract_tuple_score(answer)
          ans.append((tup,extract_feedback(answer)))
        except ValueError:
          pass

      if len(ans)>0:
        mask = (dataset['fullname'] == ids[k][0]) & (dataset['id_control'] == ids[k][1]) & (dataset['id_pregunta'] == ids[k][2])
        if mask.all() != False: print(ids[k])

        #print(ans[0])
        if ans[0][0] is not None:
           dataset.loc[mask, 'EvalGPT'] = tuple2score(ans[0][0])
           dataset.loc[mask, 'Feedback'] = ans[0][1]
