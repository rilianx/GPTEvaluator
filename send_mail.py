import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import pandas as pd
import time
import sys


# Carga tus credenciales y crea el servicio
def get_gmail_service():
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    creds = None
    # El archivo token.json almacena los tokens de acceso y actualización del usuario, y es
    # creado automáticamente cuando el flujo de autorización se completa por primera vez.
    try:
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    except FileNotFoundError:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        # Guarda las credenciales para la próxima ejecución
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def send_email(service, sender, to, subject, message_text):
    message = MIMEMultipart()
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject

    msg = MIMEText(message_text, "html")
    message.attach(msg)

    raw = base64.urlsafe_b64encode(message.as_bytes())
    raw = raw.decode()
    body = {'raw': raw}

    try:
        message = (service.users().messages().send(userId="me", body=body).execute())
        print('Message Id: %s' % message['id'])
        return message
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

# Configuración del correo
service = get_gmail_service()
sender_email = "ignacio.araya@pucv.cl"


df = pd.read_excel('resumen.xlsx')

preliminar = False

for i, row in df.iterrows():
  subject = "Revisión Control 3"
  body = ""

  if preliminar:
    body += f'Hola <b>{row["fullname"]}</b>, aquí podrás ver una <b>revisión preliminar</b> de tu control 3 realizada por GPT3.5.<p><p>'
    body += "<h3> Evaluación GPT (preliminar):</h3>"  

  else:
    body += f'Hola <b>{row["fullname"]}</b>, aquí podrás ver la <b>revisión final</b> de tu control 3.<p><p>'
    body += "<h3> Evaluación:</h3>"

  body += row["texto_concatenado"].replace('\n', '<br>')+"<p>"
  body += f'<p><b>Total de preguntas revisadas</b>: {row["revisadas"]}/8'
  body += f'<p><b>Puntaje considerando preguntas revisadas</b>:{str(row["suma_puntajes"])}/24 (para llevarlo a escala 0-100 debes multiplicar por 100/24)<p>'
  
  if preliminar:
    body += "<h3> Autoevaluación:</h3>"  
    body += "Revisa con rigurosidad tus respuestas y los comentarios y puntajes asignados por GPT. <br>En caso de <b>no estar de acuerdo</b> con alguno de los "
    body += "puntajes, por favor responde este mismo correo indicando la o las preguntas que deseas apelar.<br>"
    body += "Para cada respuesta, <b>debes explicar claramente</b> el motivo por el cual debería ser ajustada en su puntuación.<br>"
    body += "Tus argumentos <b>deben demostrar una profunda comprensión de los contenidos</b>.<br>"
    body += "Si faltó claridad en tu respuesta original, este es el momento para aclarar y profundizar en lo que quisiste decir.<br>"
    body += "Las apelaciones serán revisadas por mí (el profesor) y te informaré el puntaje definitivo.<br>"
    body += "Ten en cuenta que al realizar una apelación tu puntaje también puede bajar.<p>"
    body += "Si no apelas, de igual forma <b>tu control podría ser seleccionado</b> para ser revisado manualmente y así validar los puntajes.<p>"

  #body += "Recuerda llenar la encuesta de satisfacción para mejorar la herramienta: https://forms.gle/5hyumvjXeFtUHFw3A<p>"
  body += "Saludos,<p><p>Ignacio Araya y GPT<p>"

  email = row["email"]
  receiver_email = email
  
  print(email,"\n", body)

  # Enviar correo
  automail = False
  if service:
    send_email(service, sender_email, receiver_email, subject, body)
    print("Correo enviado a", email)
        
   
    time.sleep(1)
  
