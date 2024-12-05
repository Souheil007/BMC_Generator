from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import google.generativeai as genai  # Correct import for the generative AI library
from dotenv import load_dotenv
import re
from fuzzywuzzy import fuzz
#from mangum import Mangum

# Access your API key as an environment variable
load_dotenv()  # Add this to load the API key
api_key = os.getenv("API_KEY")  # Use your actual API key variable name from the .env file
# Set the API key for authentication
genai.configure(api_key=api_key)

app = FastAPI()


# Allow CORS for the specific origin (frontend URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to "*" only for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allow all headers (Authorization, Content-Type, etc.)
)
#handler = Mangum(app)
# Load the pre-trained model once, outside the function
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use any transformer model


#grouped_df = pd.read_pickle('concatenated_file.pkl')

def get_file_path_by_language(language: str) -> str:
    # Define file paths for each language
    file_paths = {
        "en": "grouped_df_en.pkl",
        "de": "grouped_df_de.pkl",
        "es": "grouped_df_es.pkl",
        "fr": "grouped_df_fr.pkl",
        "it": "grouped_df_it.pkl",
        "nl": "grouped_df_nl.pkl"
    }
    
    # Check if the language is supported
    if language in file_paths:
        return file_paths[language]
    else:
        # Raise an error for unsupported languages
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported languages are: {', '.join(file_paths.keys())}"
        )


def get_all_occupation_informations(occupation,grouped_df):
  # Filter the DataFrame for the occupation 'astronaut'
  occupation_description = grouped_df[grouped_df['preferredLabel1'].str.lower() == occupation.lower()]

  # Access the concatenated column for the astronaut
  occupation_description_concatenated = occupation_description['concatenated'].values[0]  # Get the value of the concatenated column

  # Print or use the concatenated result
  return occupation_description_concatenated


def find_top_matching_occupations(file_path: str, user_input: str, top_n: int = 3):
    # Load the file from the given file path
    grouped_df = pd.read_pickle(file_path)
    
    # Encode the user input
    user_input_embedding = model.encode(user_input)

    # Compute similarity scores
    grouped_df['similarity'] = grouped_df['description_embedding'].apply(
        lambda emb: cosine_similarity([user_input_embedding], [emb]).item()
    )

    # Sort by similarity and select the top N matches
    top_matches = grouped_df.nlargest(top_n, 'similarity')

    # Store matched occupation titles in a list and concatenate for display
    matched_occupations_str = ""
    matched_occupations_list = []
    
    for idx, match in top_matches.iterrows():
        occupation = match['preferredLabel1']
        print(f"Occupation: {occupation}")
        print(f"Description: {match['description1']}")
        print(f"Similarity score: {match['similarity']}")
        print("\n---\n")
        
        # Add to string and list
        matched_occupations_str += occupation + ", "
        matched_occupations_list.append(occupation)

    # Return the concatenated string and list of top matches
    return matched_occupations_str, matched_occupations_list

#because in german language we have occupations like friseur/friseurin so we will map this occupation parts to it , exp : map[firseur]=friseur/friseurin
def construct_dict_from_list(occupations_list):
    # Initialize an empty dictionary
    occupation_dict = {}
    
    # Iterate through the list of occupations
    for occupation in occupations_list:
        # Split the occupation into individual words using "/"
        parts = occupation.split("/")
        
        # Map each part back to the original occupation
        for part in parts:
            occupation_dict[part.strip()] = occupation
    
    return occupation_dict

def ask_AI(user_idea: str, matched_occupations: str, language: str):
    # Define the prompt based on the language
    prompts = {
        "en": (
            f"Based on the user's idea: {user_idea}, "
            f"if any occupation from this list: {matched_occupations} matches the user's idea, "
            f"return that occupation; otherwise, return 'no'. This should be on the first line. "
            f"Following that, give me a 300-word paragraph on the skills the user should have to launch their business. "
            f"On the last line, return job vacancies the user should post for their business."
        ),
        "de": (
            f"Basierend auf der Idee des Benutzers: {user_idea}, "
            f"wenn eine der Berufe aus dieser Liste: {matched_occupations} zur Idee des Benutzers passt, "
            f"geben Sie diesen Beruf zurück; andernfalls geben Sie 'nein' zurück. Dies sollte in der ersten Zeile stehen. "
            f"Geben Sie mir anschließend einen 300-Wörter-Absatz über die Fähigkeiten, die der Benutzer haben sollte, um sein Unternehmen zu gründen. "
            f"Geben Sie in der letzten Zeile Stellenanzeigen zurück, die der Benutzer für sein Unternehmen veröffentlichen sollte."
        ),
        "es": (
            f"Basado en la idea del usuario: {user_idea}, "
            f"si alguna ocupación de esta lista: {matched_occupations} coincide con la idea del usuario, "
            f"devuelva esa ocupación; de lo contrario, devuelva 'no'. Esto debe estar en la primera línea. "
            f"A continuación, dame un párrafo de 300 palabras sobre las habilidades que el usuario debería tener para lanzar su negocio. "
            f"En la última línea, devuelve las vacantes laborales que el usuario debería publicar para su negocio."
        ),
        "fr": (
            f"En se basant sur l'idée de l'utilisateur : {user_idea}, "
            f"si une profession de cette liste : {matched_occupations} correspond à l'idée de l'utilisateur, "
            f"retournez cette profession ; sinon, retournez 'non'. Cela doit être sur la première ligne. "
            f"Ensuite, donnez-moi un paragraphe de 300 mots sur les compétences que l'utilisateur devrait avoir pour lancer son entreprise. "
            f"À la dernière ligne, retournez les offres d'emploi que l'utilisateur devrait publier pour son entreprise."
        ),
        "it": (
            f"Basandosi sull'idea dell'utente: {user_idea}, "
            f"se una qualsiasi occupazione da questo elenco: {matched_occupations} corrisponde all'idea dell'utente, "
            f"restituire tale occupazione; altrimenti, restituire 'no'. Questo dovrebbe essere sulla prima riga. "
            f"Successivamente, forniscimi un paragrafo di 300 parole sulle competenze che l'utente dovrebbe avere per avviare la propria attività. "
            f"Sull'ultima riga, restituisci le offerte di lavoro che l'utente dovrebbe pubblicare per la sua attività."
        ),
        "nl": (
            f"Op basis van het idee van de gebruiker: {user_idea}, "
            f"als een beroep uit deze lijst: {matched_occupations} overeenkomt met het idee van de gebruiker, "
            f"geef dat beroep terug; anders geef 'nee' terug. Dit moet op de eerste regel staan. "
            f"Geef me vervolgens een alinea van 300 woorden over de vaardigheden die de gebruiker zou moeten hebben om zijn bedrijf te starten. "
            f"Geef op de laatste regel de vacatures terug die de gebruiker voor zijn bedrijf zou moeten plaatsen."
        ),
    }

    # Check if the language is supported
    if language in prompts:
        prompt= prompts[language]
    else:
        # Raise an error for unsupported languages
        raise ValueError(
            f"Unsupported language: {language}. Supported languages are: {', '.join(prompts.keys())}"
        )


    # Call the model to generate content
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    print(response.text)
    
    # Extract the generated response text
    response_text = response.text

    # Split the response to get occupation match and skills paragraph separately
    parts = response_text.split('\n', 1)  # Split by first newline
    occupation_match = parts[0].replace("Occupation Match: ", "").strip()
    skills_paragraph = parts[1].strip() if len(parts) > 1 else "No skills information provided."
    
    # Attempt to clean and parse the occupation match
    occupation_match_re = re.search(r"(Occupation Match:|Matching Occupation:|\*\*Matching Occupation:\*\*|^\*\*(.+)\*\*$)\s*(.*)", occupation_match)
    if occupation_match_re:
        occupation_match = occupation_match_re.group(2).strip()
    else:
        occupation_match = occupation_match  # Fallback to the raw second line

    return  occupation_match, skills_paragraph


def generate_content(user_idea, occupation_match, skills_paragraph, get_all_occupation_informations,matched_occupations_list,language,grouped_df):
    # Ensure all variables are strings or have default values
    user_idea = user_idea if user_idea is not None else ""
    skills_paragraph = skills_paragraph if skills_paragraph is not None else ""
    occupation_match = occupation_match if occupation_match is not None else ""

    # Construct the content based on whether an occupation matches or not
    if occupation_match.lower() == "no" or occupation_match == "" or occupation_match.lower() not in matched_occupations_list:
        if language == "en":
            # English content for no specific match
            content = (
                f"The user's idea: '{user_idea}'.\n\n"
                "Although no exact occupation match was found, here is a 300-word summary of essential skills "
                "that would help the user succeed in this business:\n\n" + skills_paragraph
            )
        elif language == "de":
            # German content for no specific match
            content = (
                f"Die Idee des Benutzers: '{user_idea}'.\n\n"
                "Obwohl keine genaue Berufsübereinstimmung gefunden wurde, finden Sie hier eine 300-Wörter-Zusammenfassung "
                "der wesentlichen Fähigkeiten, die dem Benutzer helfen würden, in diesem Geschäft erfolgreich zu sein:\n\n" + skills_paragraph
            )
        elif language == "es":
            # Spanish content for no specific match
            content = (
                f"La idea del usuario: '{user_idea}'.\n\n"
                "Aunque no se encontró una coincidencia exacta de ocupación, aquí hay un resumen de 300 palabras de las habilidades esenciales "
                "que ayudarían al usuario a tener éxito en este negocio:\n\n" + skills_paragraph
            )
        elif language == "fr":
            # French content for no specific match
            content = (
                f"L'idée de l'utilisateur : '{user_idea}'.\n\n"
                "Bien qu'aucune correspondance exacte de métier n'ait été trouvée, voici un résumé de 300 mots des compétences essentielles "
                "qui aideraient l'utilisateur à réussir dans cette activité :\n\n" + skills_paragraph
            )
        elif language == "it":
            # Italian content for no specific match
            content = (
                f"L'idea dell'utente: '{user_idea}'.\n\n"
                "Sebbene non sia stata trovata una corrispondenza esatta con un'occupazione, ecco un riepilogo di 300 parole delle competenze essenziali "
                "che aiuterebbero l'utente a avere successo in questo settore:\n\n" + skills_paragraph
            )
        elif language == "nl":
            # Dutch content for no specific match
            content = (
                f"Het idee van de gebruiker: '{user_idea}'.\n\n"
                "Hoewel er geen exacte beroepsmatch is gevonden, volgt hier een samenvatting van 300 woorden over de essentiële vaardigheden "
                "die de gebruiker zouden helpen succesvol te zijn in dit bedrijf:\n\n" + skills_paragraph
            )
        else:
            raise ValueError(f"Unsupported language: {language}. Supported languages are 'en', 'de', 'es', 'fr', 'it', 'nl'.")
    else:
        # Content when a specific occupation matches
        occu_infos = get_all_occupation_informations(occupation_match, grouped_df) or ""
        if language == "en":
            # English content for a matched occupation
            content = (
                f"The user's idea: '{user_idea}'.\n\n"
                f"The best matching occupation is: '{occupation_match}'.\n\n"
                "Here is a summary of the skills needed to launch a business in this field:\n\n" + skills_paragraph +
                "\n\nAdditionally, here are relevant details about the occupation:\n\n" + occu_infos
            )
        elif language == "de":
            # German content for a matched occupation
            content = (
                f"Die Idee des Benutzers: '{user_idea}'.\n\n"
                f"Der am besten passende Beruf ist: '{occupation_match}'.\n\n"
                "Hier ist eine Zusammenfassung der Fähigkeiten, die erforderlich sind, um ein Unternehmen in diesem Bereich zu gründen:\n\n" + skills_paragraph +
                "\n\nZusätzlich finden Sie hier relevante Details zu diesem Beruf:\n\n" + occu_infos
            )
        elif language == "es":
            # Spanish content for a matched occupation
            content = (
                f"La idea del usuario: '{user_idea}'.\n\n"
                f"La ocupación que mejor se adapta es: '{occupation_match}'.\n\n"
                "Aquí hay un resumen de las habilidades necesarias para iniciar un negocio en este campo:\n\n" + skills_paragraph +
                "\n\nAdemás, aquí hay detalles relevantes sobre la ocupación:\n\n" + occu_infos
            )
        elif language == "fr":
            # French content for a matched occupation
            content = (
                f"L'idée de l'utilisateur : '{user_idea}'.\n\n"
                f"La profession la mieux adaptée est : '{occupation_match}'.\n\n"
                "Voici un résumé des compétences nécessaires pour lancer une entreprise dans ce domaine :\n\n" + skills_paragraph +
                "\n\nDe plus, voici des informations pertinentes sur la profession :\n\n" + occu_infos
            )
        elif language == "it":
            # Italian content for a matched occupation
            content = (
                f"L'idea dell'utente: '{user_idea}'.\n\n"
                f"L'occupazione che si adatta meglio è: '{occupation_match}'.\n\n"
                "Ecco un riepilogo delle competenze necessarie per avviare un'attività in questo settore:\n\n" + skills_paragraph +
                "\n\nInoltre, ecco i dettagli rilevanti sull'occupazione:\n\n" + occu_infos
            )
        elif language == "nl":
            # Dutch content for a matched occupation
            content = (
                f"Het idee van de gebruiker: '{user_idea}'.\n\n"
                f"Het best bijpassende beroep is: '{occupation_match}'.\n\n"
                "Hier is een samenvatting van de vaardigheden die nodig zijn om een bedrijf in dit veld te starten:\n\n" + skills_paragraph +
                "\n\nDaarnaast zijn hier relevante details over het beroep:\n\n" + occu_infos
            )
        else:
            raise ValueError(f"Unsupported language: {language}. Supported languages are 'en', 'de', 'es', 'fr', 'it', 'nl'.")

    return content

def process_full_BMC(content,language):
    if language.lower() == "en" or language.lower() == "english":
        prompt = (
        "Generate a comprehensive Business Model Canvas (BMC) for the following role using the provided description"
        "Please provide detailed paragraphs for each of the following sections: "
        "'Customer Segments', 'Value Proposition', 'Customer Relationships', 'Channels', 'Revenue Streams', "
        "'Key Resources', 'Key Activities', 'Key Partners', and 'Cost Structure'.\n\n"
        "Role Description:\n" + content + "\n\n"

        # Customer Segments
        "Customer Segments: Who are the customers? What are their key needs, behaviors, and motivations? "
        "How do they interact with or benefit from this role? What challenges or pain points are they facing, "
        "and how does this role help address those issues?\n\n"
        "give me a 300 word paragraph higlighting all the customer segments"
        "Provide a detailed paragraph, not bullet points, outlining customer segments."
        "Please ensure the customer segments are well-defined and aligned with the context of this role."

        # Value Proposition
        "Value Proposition: What unique value does this role bring to customers? How does this role solve customers' "
        "problems or fulfill their needs? What benefits or advantages does it offer? How does it differentiate itself "
        "from competitors?\n\n"
        "Provide a detailed paragraph, not bullet points, outlining the core value proposition."
        "give me a 300 word paragraph higlighting all the value propositions"
        "Please ensure the value propositions are clearly defined and aligned with the context of this role."

        # Customer Relationships
        "Customer Relationships: What type of relationship is established and maintained with customers? "
        "Does this role involve personal assistance, self-service, or automation in building customer relationships? "
        "How does this role ensure long-term customer satisfaction?\n\n"
        "Provide a detailed paragraph, not bullet points, explaining the approach to customer relationships."
        "give me a 300 word paragraph higlighting all the customer relationships"
        "Please ensure the customer relationships are clearly defined and aligned with the context of this role."

        # Channels
        "Channels: What are the primary channels through which this role delivers its value to customers? How does it reach customers, "
        "and what methods or platforms are used to communicate, distribute, or provide services?\n\n"
        "Provide a detailed paragraph, not bullet points, describing the role's channels of distribution and communication."
        "give me a 300 word paragraph highlighting all the channels"
        "Please ensure the channels are clearly defined and aligned with the context of this role."

        # Revenue Streams
        "Revenue Streams: How does this role generate revenue? What are the different ways in which it brings in money? "
        "Does it rely on direct sales, subscriptions, licensing, or other methods of generating revenue?\n\n"
        "Provide a detailed paragraph, not bullet points, explaining how this role contributes to the organization's revenue streams."
        "give me a 300 word paragraph highlighting all the revenue streams"
        "Please ensure the revenue streams are clearly defined and aligned with the context of this role."

        # Key Resources
        "Key Resources: What are the essential resources needed for this role to deliver its value proposition? "
        "This could include human resources, physical assets, intellectual property, or financial resources.\n\n"
        "Provide a detailed paragraph, not bullet points, explaining the key resources necessary for this role's success."
        "give me a 300 word paragraph highlighting all the key ressources"
        "Please ensure the key resources are clearly defined and aligned with the context of this role."

        # Key Activities
        "Key Activities: What are the core activities that this role needs to perform to deliver its value proposition? "
        "This includes the main tasks and responsibilities that are essential for success.\n\n"
        "Provide a detailed paragraph, not bullet points, explaining the key activities involved in this role."
        "give me job names for my buisness so i can post vacancies"
        "give me a 300 word paragraph highlighting all the key activities"
        "Please ensure the key activities are clearly defined and aligned with the context of this role."

        # Key Partners
        "Key Partners: Who are the main partners or collaborators this role interacts with? "
        "This could include suppliers, alliances, or other stakeholders that are crucial for success.\n\n"
        "Provide a detailed paragraph, not bullet points, explaining the key partners involved in this role."
        "give me a 300 word paragraph highlighting all the key partners"
        "Please ensure the key partners are clearly defined and aligned with the context of this role."

        # Cost Structure
        "Cost Structure: What are the primary costs related to this role? This could include salaries, operational costs, resources, "
        "and other expenses. How do these costs relate to the key resources, activities, and partnerships?\n\n"
        "Provide a detailed paragraph, not bullet points, explaining the cost structure for this role."
        "give me a 300 word paragraph highlighting all the cost structures"
        "Please ensure the cost structure is clearly defined and aligned with the context of this role."

        "each section must be a 300 word paragraph"
        "no bullet points in the sections i want a paragraph"
        "Please provide a detailed response for each section."
        "every segment is a paragraph !"
        )
    elif language.lower() == "es" or language.lower() == "espagnol":
        prompt = (
        "Genera un Business Model Canvas (BMC) completo para el siguiente rol utilizando la descripción proporcionada. "
        "Por favor, proporciona párrafos detallados para cada una de las siguientes secciones: "
        "'Segmentos de clientes', 'Propuesta de valor', 'Relaciones con los clientes', 'Canales', 'Flujos de ingresos', "
        "'Recursos clave', 'Actividades clave', 'Socios clave' y 'Estructura de costos'.\n\n"
        "Descripción del rol:\n" + content + "\n\n"

        # Segmentos de clientes
        "Segmentos de clientes: ¿Quiénes son los clientes? ¿Cuáles son sus necesidades, comportamientos y motivaciones clave? "
        "¿Cómo interactúan con este rol o se benefician de él? ¿Qué desafíos o puntos débiles enfrentan y cómo este rol ayuda a resolver esos problemas?\n\n"
        "Dame un párrafo de 300 palabras destacando todos los segmentos de clientes. "
        "Proporciona un párrafo detallado, sin usar puntos, describiendo los segmentos de clientes.\n\n"

        # Propuesta de valor
        "Propuesta de valor: ¿Qué valor único aporta este rol a los clientes? ¿Cómo resuelve este rol los problemas de los clientes o satisface sus necesidades? "
        "¿Qué beneficios o ventajas ofrece? ¿Cómo se diferencia de los competidores?\n\n"
        "Dame un párrafo de 300 palabras destacando todas las propuestas de valor. "
        "Proporciona un párrafo detallado, sin usar puntos, describiendo la propuesta de valor principal.\n\n"

        # Relaciones con los clientes
        "Relaciones con los clientes: ¿Qué tipo de relación se establece y mantiene con los clientes? "
        "¿Este rol implica asistencia personal, autoservicio o automatización para construir relaciones con los clientes? "
        "¿Cómo garantiza este rol la satisfacción del cliente a largo plazo?\n\n"
        "Dame un párrafo de 300 palabras destacando todas las relaciones con los clientes. "
        "Proporciona un párrafo detallado, sin usar puntos, explicando el enfoque de las relaciones con los clientes.\n\n"

        # Canales
        "Canales: ¿Cuáles son los canales principales a través de los cuales este rol entrega su valor a los clientes? ¿Cómo llega a los clientes, "
        "y qué métodos o plataformas se utilizan para comunicar, distribuir o proporcionar servicios?\n\n"
        "Dame un párrafo de 300 palabras destacando todos los canales. "
        "Proporciona un párrafo detallado, sin usar puntos, describiendo los canales de distribución y comunicación del rol.\n\n"

        # Flujos de ingresos
        "Flujos de ingresos: ¿Cómo genera ingresos este rol? ¿Cuáles son las diferentes formas en que genera dinero? "
        "¿Depende de ventas directas, suscripciones, licencias u otros métodos para generar ingresos?\n\n"
        "Dame un párrafo de 300 palabras destacando todos los flujos de ingresos. "
        "Proporciona un párrafo detallado, sin usar puntos, explicando cómo este rol contribuye a los flujos de ingresos de la organización.\n\n"

        # Recursos clave
        "Recursos clave: ¿Cuáles son los recursos esenciales necesarios para que este rol entregue su propuesta de valor? "
        "Esto podría incluir recursos humanos, activos físicos, propiedad intelectual o recursos financieros.\n\n"
        "Dame un párrafo de 300 palabras destacando todos los recursos clave. "
        "Proporciona un párrafo detallado, sin usar puntos, explicando los recursos clave necesarios para el éxito de este rol.\n\n"

        # Actividades clave
        "Actividades clave: ¿Cuáles son las actividades principales que este rol necesita realizar para entregar su propuesta de valor? "
        "Esto incluye las tareas y responsabilidades esenciales para el éxito.\n\n"
        "Dame un párrafo de 300 palabras destacando todas las actividades clave. "
        "Proporciona un párrafo detallado, sin usar puntos, explicando las actividades clave involucradas en este rol.\n\n"

        # Socios clave
        "Socios clave: ¿Quiénes son los principales socios o colaboradores con los que interactúa este rol? "
        "Esto podría incluir proveedores, alianzas u otras partes interesadas cruciales para el éxito.\n\n"
        "Dame un párrafo de 300 palabras destacando todos los socios clave. "
        "Proporciona un párrafo detallado, sin usar puntos, explicando los socios clave involucrados en este rol.\n\n"

        # Estructura de costos
        "Estructura de costos: ¿Cuáles son los principales costos relacionados con este rol? Esto podría incluir salarios, costos operativos, recursos "
        "y otros gastos. ¿Cómo se relacionan estos costos con los recursos clave, actividades y asociaciones?\n\n"
        "Dame un párrafo de 300 palabras destacando todas las estructuras de costos. "
        "Proporciona un párrafo detallado, sin usar puntos, explicando la estructura de costos para este rol.\n\n"

        "Cada sección debe ser un párrafo de 300 palabras. No se deben usar puntos en las secciones, solo párrafos. "
        "Por favor, proporciona una respuesta detallada para cada sección. ¡Cada sección es un párrafo!"
        )
    elif language.lower() == "it" or language.lower() == "italien":
        prompt = (
        "Genera un Business Model Canvas (BMC) completo per il seguente ruolo utilizzando la descrizione fornita. "
        "Si prega di fornire paragrafi dettagliati per ciascuna delle seguenti sezioni: "
        "'Segmenti di clientela', 'Proposta di valore', 'Relazioni con i clienti', 'Canali', 'Flussi di entrate', "
        "'Risorse chiave', 'Attività chiave', 'Partner chiave' e 'Struttura dei costi'.\n\n"
        "Descrizione del ruolo:\n" + content + "\n\n"

        # Segmenti di clientela
        "Segmenti di clientela: Chi sono i clienti? Quali sono i loro bisogni, comportamenti e motivazioni principali? "
        "Come interagiscono con questo ruolo o come ne beneficiano? Quali sfide o problemi affrontano e come questo ruolo aiuta a risolverli?\n\n"
        "Dammi un paragrafo di 300 parole evidenziando tutti i segmenti di clientela. "
        "Fornisci un paragrafo dettagliato, senza usare punti elenco, che descriva i segmenti di clientela.\n\n"

        # Proposta di valore
        "Proposta di valore: Quale valore unico porta questo ruolo ai clienti? Come risolve questo ruolo i problemi dei clienti o soddisfa i loro bisogni? "
        "Quali benefici o vantaggi offre? Come si differenzia dalla concorrenza?\n\n"
        "Dammi un paragrafo di 300 parole evidenziando tutte le proposte di valore. "
        "Fornisci un paragrafo dettagliato, senza usare punti elenco, che descriva la proposta di valore principale.\n\n"

        # Relazioni con i clienti
        "Relazioni con i clienti: Che tipo di relazione si stabilisce e mantiene con i clienti? "
        "Questo ruolo implica assistenza personale, autoservizio o automazione per costruire relazioni con i clienti? "
        "Come garantisce questo ruolo la soddisfazione del cliente a lungo termine?\n\n"
        "Dammi un paragrafo di 300 parole evidenziando tutte le relazioni con i clienti. "
        "Fornisci un paragrafo dettagliato, senza usare punti elenco, che spieghi l'approccio alle relazioni con i clienti.\n\n"

        # Canali
        "Canali: Quali sono i principali canali attraverso i quali questo ruolo consegna il suo valore ai clienti? Come raggiunge i clienti, "
        "e quali metodi o piattaforme vengono utilizzati per comunicare, distribuire o fornire i servizi?\n\n"
        "Dammi un paragrafo di 300 parole evidenziando tutti i canali. "
        "Fornisci un paragrafo dettagliato, senza usare punti elenco, che descriva i canali di distribuzione e comunicazione del ruolo.\n\n"

        # Flussi di entrate
        "Flussi di entrate: Come genera entrate questo ruolo? Quali sono i diversi modi in cui porta denaro? "
        "Dipende dalle vendite dirette, abbonamenti, licenze o altri metodi per generare entrate?\n\n"
        "Dammi un paragrafo di 300 parole evidenziando tutti i flussi di entrate. "
        "Fornisci un paragrafo dettagliato, senza usare punti elenco, che spieghi come questo ruolo contribuisce ai flussi di entrate dell'organizzazione.\n\n"

        # Risorse chiave
        "Risorse chiave: Quali sono le risorse essenziali necessarie affinché questo ruolo consegni la sua proposta di valore? "
        "Queste potrebbero includere risorse umane, beni fisici, proprietà intellettuale o risorse finanziarie.\n\n"
        "Dammi un paragrafo di 300 parole evidenziando tutte le risorse chiave. "
        "Fornisci un paragrafo dettagliato, senza usare punti elenco, che spieghi le risorse chiave necessarie per il successo di questo ruolo.\n\n"

        # Attività chiave
        "Attività chiave: Quali sono le attività principali che questo ruolo deve eseguire per consegnare la sua proposta di valore? "
        "Queste includono i compiti e le responsabilità essenziali per il successo.\n\n"
        "Dammi un paragrafo di 300 parole evidenziando tutte le attività chiave. "
        "Fornisci un paragrafo dettagliato, senza usare punti elenco, che spieghi le attività chiave coinvolte in questo ruolo.\n\n"

        # Partner chiave
        "Partner chiave: Chi sono i principali partner o collaboratori con cui questo ruolo interagisce? "
        "Questi potrebbero includere fornitori, alleanze o altre parti interessate cruciali per il successo.\n\n"
        "Dammi un paragrafo di 300 parole evidenziando tutti i partner chiave. "
        "Fornisci un paragrafo dettagliato, senza usare punti elenco, che spieghi i partner chiave coinvolti in questo ruolo.\n\n"

        # Struttura dei costi
        "Struttura dei costi: Quali sono i principali costi relativi a questo ruolo? Questi potrebbero includere stipendi, costi operativi, risorse "
        "e altre spese. Come si relazionano questi costi con le risorse chiave, le attività e le partnership?\n\n"
        "Dammi un paragrafo di 300 parole evidenziando tutte le strutture dei costi. "
        "Fornisci un paragrafo dettagliato, senza usare punti elenco, che spieghi la struttura dei costi per questo ruolo.\n\n"

        "Ogni sezione deve essere un paragrafo di 300 parole. Non devono essere utilizzati punti elenco nelle sezioni, solo paragrafi. "
        "Si prega di fornire una risposta dettagliata per ogni sezione. Ogni sezione è un paragrafo!"
        )
    elif language.lower() == "nl" or language.lower() == "dutch":
        prompt = (
        "Genereer een uitgebreid Business Model Canvas (BMC) voor de volgende rol op basis van de gegeven beschrijving. "
        "Geef gedetailleerde paragrafen voor elk van de volgende secties: "
        "'Klantsegmenten', 'Waardepropositie', 'Klantrelaties', 'Kanalen', 'Inkomstenstromen', "
        "'Key Resources', 'Key Activities', 'Key Partners' en 'Kostenstructuur'.\n\n"
        "Rolbeschrijving:\n" + content + "\n\n"

        # Klantsegmenten
        "Klantsegmenten: Wie zijn de klanten? Wat zijn hun belangrijkste behoeften, gedragingen en motivaties? "
        "Hoe interacteert deze rol met de klanten of hoe profiteren ze van deze rol? Welke uitdagingen of pijnpunten hebben ze, "
        "en hoe helpt deze rol om deze problemen op te lossen?\n\n"
        "Geef een paragraaf van 300 woorden die alle klantsegmenten beschrijft. "
        "Geef een gedetailleerde paragraaf zonder opsommingstekens die de klantsegmenten uitlegt.\n\n"

        # Waardepropositie
        "Waardepropositie: Welke unieke waarde biedt deze rol aan klanten? Hoe lost deze rol de problemen van klanten op of vervult het hun behoeften? "
        "Welke voordelen of voordelen biedt het? Hoe onderscheidt het zich van concurrenten?\n\n"
        "Geef een paragraaf van 300 woorden die alle waardeproposities beschrijft. "
        "Geef een gedetailleerde paragraaf zonder opsommingstekens die de belangrijkste waardepropositie uitlegt.\n\n"

        # Klantrelaties
        "Klantrelaties: Wat voor soort relatie wordt er opgebouwd en onderhouden met de klanten? "
        "Betrekt deze rol persoonlijke assistentie, zelfbediening of automatisering om klantrelaties op te bouwen? "
        "Hoe zorgt deze rol voor langdurige klanttevredenheid?\n\n"
        "Geef een paragraaf van 300 woorden die alle klantrelaties beschrijft. "
        "Geef een gedetailleerde paragraaf zonder opsommingstekens die de benadering van klantrelaties uitlegt.\n\n"

        # Kanalen
        "Kanalen: Wat zijn de belangrijkste kanalen waarmee deze rol zijn waarde aan klanten levert? Hoe bereikt deze rol de klanten, "
        "en welke methoden of platforms worden gebruikt om te communiceren, te distribueren of diensten te leveren?\n\n"
        "Geef een paragraaf van 300 woorden die alle kanalen beschrijft. "
        "Geef een gedetailleerde paragraaf zonder opsommingstekens die de kanalen voor distributie en communicatie uitlegt.\n\n"

        # Inkomstenstromen
        "Inkomstenstromen: Hoe genereert deze rol inkomsten? Wat zijn de verschillende manieren waarop het geld genereert? "
        "Is het afhankelijk van directe verkoop, abonnementen, licenties of andere methoden om inkomsten te genereren?\n\n"
        "Geef een paragraaf van 300 woorden die alle inkomstenstromen beschrijft. "
        "Geef een gedetailleerde paragraaf zonder opsommingstekens die uitlegt hoe deze rol bijdraagt aan de inkomstenstromen van de organisatie.\n\n"

        # Key Resources
        "Key Resources: Wat zijn de essentiële middelen die deze rol nodig heeft om zijn waardepropositie te leveren? "
        "Dit kan personeelsmiddelen, fysieke activa, intellectuele eigendom of financiële middelen omvatten.\n\n"
        "Geef een paragraaf van 300 woorden die alle key resources beschrijft. "
        "Geef een gedetailleerde paragraaf zonder opsommingstekens die de belangrijke middelen uitlegt die nodig zijn voor het succes van deze rol.\n\n"

        # Key Activities
        "Key Activities: Wat zijn de belangrijkste activiteiten die deze rol moet uitvoeren om zijn waardepropositie te leveren? "
        "Dit omvat de belangrijkste taken en verantwoordelijkheden die essentieel zijn voor succes.\n\n"
        "Geef een paragraaf van 300 woorden die alle key activities beschrijft. "
        "Geef een gedetailleerde paragraaf zonder opsommingstekens die de belangrijke activiteiten van deze rol uitlegt.\n\n"

        # Key Partners
        "Key Partners: Wie zijn de belangrijkste partners of samenwerkingsverbanden waarmee deze rol interactie heeft? "
        "Dit kunnen leveranciers, allianties of andere belanghebbenden zijn die cruciaal zijn voor succes.\n\n"
        "Geef een paragraaf van 300 woorden die alle key partners beschrijft. "
        "Geef een gedetailleerde paragraaf zonder opsommingstekens die de belangrijkste partners uitlegt die betrokken zijn bij deze rol.\n\n"

        # Kostenstructuur
        "Kostenstructuur: Wat zijn de belangrijkste kosten die verband houden met deze rol? Dit kan lonen, operationele kosten, middelen "
        "en andere uitgaven omvatten. Hoe verhouden deze kosten zich tot de key resources, activiteiten en partnerschappen?\n\n"
        "Geef een paragraaf van 300 woorden die de kostenstructuur beschrijft. "
        "Geef een gedetailleerde paragraaf zonder opsommingstekens die de kostenstructuur voor deze rol uitlegt.\n\n"

        "Elke sectie moet een paragraaf van 300 woorden zijn. Gebruik geen opsommingstekens in de secties, alleen paragrafen. "
        "Geef gedetailleerde antwoorden voor elke sectie. Elke sectie is een paragraaf!"
        )
    elif language.lower() == "fr" or language.lower() == "francais":
        prompt = (
        "Générez un Business Model Canvas (BMC) complet pour le rôle suivant en utilisant la description fournie. "
        "Veuillez fournir des paragraphes détaillés pour chacune des sections suivantes : "
        "'Segments de clientèle', 'Proposition de valeur', 'Relations avec les clients', 'Canaux', 'Sources de revenus', "
        "'Ressources clés', 'Activités clés', 'Partenaires clés' et 'Structure des coûts'.\n\n"
        "Description du rôle :\n" + content + "\n\n"

        # Segments de clientèle
        "Segments de clientèle : Qui sont les clients ? Quels sont leurs besoins, comportements et motivations clés ? "
        "Comment interagissent-ils avec ce rôle ou en bénéficient-ils ? Quels défis ou points de douleur rencontrent-ils, "
        "et comment ce rôle contribue-t-il à résoudre ces problèmes ?\n\n"
        "Donnez-moi un paragraphe de 300 mots mettant en avant tous les segments de clientèle. "
        "Fournissez un paragraphe détaillé, sans utiliser de points, décrivant les segments de clientèle.\n\n"

        # Proposition de valeur
        "Proposition de valeur : Quelle valeur unique ce rôle apporte-t-il aux clients ? Comment ce rôle résout-il les problèmes "
        "des clients ou répond-il à leurs besoins ? Quels avantages ou bénéfices offre-t-il ? Comment se différencie-t-il "
        "de la concurrence ?\n\n"
        "Donnez-moi un paragraphe de 300 mots mettant en avant toutes les propositions de valeur. "
        "Fournissez un paragraphe détaillé, sans utiliser de points, décrivant la proposition de valeur principale.\n\n"

        # Relations avec les clients
        "Relations avec les clients : Quel type de relation est établi et maintenu avec les clients ? "
        "Ce rôle implique-t-il une assistance personnelle, un libre-service ou une automatisation pour construire des relations avec les clients ? "
        "Comment ce rôle garantit-il la satisfaction des clients sur le long terme ?\n\n"
        "Donnez-moi un paragraphe de 300 mots mettant en avant toutes les relations avec les clients. "
        "Fournissez un paragraphe détaillé, sans utiliser de points, expliquant l'approche des relations avec les clients.\n\n"

        # Canaux
        "Canaux : Quels sont les principaux canaux par lesquels ce rôle délivre sa valeur aux clients ? Comment ce rôle atteint-il les clients, "
        "et quelles méthodes ou plateformes sont utilisées pour communiquer, distribuer ou fournir des services ?\n\n"
        "Donnez-moi un paragraphe de 300 mots mettant en avant tous les canaux. "
        "Fournissez un paragraphe détaillé, sans utiliser de points, décrivant les canaux de distribution et de communication du rôle.\n\n"

        # Sources de revenus
        "Sources de revenus : Comment ce rôle génère-t-il des revenus ? Quels sont les différents moyens par lesquels il génère de l'argent ? "
        "S'appuie-t-il sur des ventes directes, des abonnements, des licences ou d'autres méthodes pour générer des revenus ?\n\n"
        "Donnez-moi un paragraphe de 300 mots mettant en avant toutes les sources de revenus. "
        "Fournissez un paragraphe détaillé, sans utiliser de points, expliquant comment ce rôle contribue aux sources de revenus de l'organisation.\n\n"

        # Ressources clés
        "Ressources clés : Quelles sont les ressources essentielles nécessaires à ce rôle pour fournir sa proposition de valeur ? "
        "Cela peut inclure des ressources humaines, des actifs physiques, de la propriété intellectuelle ou des ressources financières.\n\n"
        "Donnez-moi un paragraphe de 300 mots mettant en avant toutes les ressources clés. "
        "Fournissez un paragraphe détaillé, sans utiliser de points, expliquant les ressources clés nécessaires au succès de ce rôle.\n\n"

        # Activités clés
        "Activités clés : Quelles sont les activités principales que ce rôle doit accomplir pour fournir sa proposition de valeur ? "
        "Cela inclut les tâches et responsabilités essentielles à son succès.\n\n"
        "Donnez-moi un paragraphe de 300 mots mettant en avant toutes les activités clés. "
        "Fournissez un paragraphe détaillé, sans utiliser de points, expliquant les activités clés impliquées dans ce rôle.\n\n"

        # Partenaires clés
        "Partenaires clés : Quels sont les principaux partenaires ou collaborateurs avec lesquels ce rôle interagit ? "
        "Cela peut inclure des fournisseurs, des alliances ou d'autres parties prenantes essentielles à son succès.\n\n"
        "Donnez-moi un paragraphe de 300 mots mettant en avant tous les partenaires clés. "
        "Fournissez un paragraphe détaillé, sans utiliser de points, expliquant les partenaires clés impliqués dans ce rôle.\n\n"

        # Structure des coûts
        "Structure des coûts : Quels sont les principaux coûts liés à ce rôle ? Cela peut inclure les salaires, les coûts opérationnels, les ressources "
        "et d'autres dépenses. Comment ces coûts se rapportent-ils aux ressources clés, activités et partenariats ?\n\n"
        "Donnez-moi un paragraphe de 300 mots mettant en avant toutes les structures de coûts. "
        "Fournissez un paragraphe détaillé, sans utiliser de points, expliquant la structure des coûts pour ce rôle.\n\n"

        "Chaque section doit être un paragraphe de 300 mots. Pas de points dans les sections, uniquement des paragraphes. "
        "Veuillez fournir une réponse détaillée pour chaque section. Chaque section est un paragraphe !"
        )
    else :
        prompt = (
        "Erstellen Sie eine umfassende Business Model Canvas (BMC) für die folgende Rolle basierend auf der bereitgestellten Beschreibung."
        "Bitte erstellen Sie detaillierte Absätze für jeden der folgenden Abschnitte: "
        "'Kundensegmente', 'Wertangebote', 'Kundenbeziehungen', 'Kanäle', 'Einnahmequellen', "
        "'Schlüsselressourcen', 'Schlüsselaktivitäten', 'Schlüsselpartner' und 'Kostenstruktur'.\n\n"
        "Rollenbeschreibung:\n" + content + "\n\n"

        # Kundensegmente
        "Kundensegmente: Wer sind die Kunden? Was sind ihre wichtigsten Bedürfnisse, Verhaltensweisen und Motivationen? "
        "Wie interagieren sie mit dieser Rolle oder profitieren von ihr? Welche Herausforderungen oder Probleme haben sie, "
        "und wie hilft diese Rolle, diese zu lösen?\n\n"
        "Erstellen Sie einen 300 Wörter langen Absatz, der alle Kundensegmente hervorhebt."
        "Liefern Sie einen detaillierten Absatz, keine Aufzählungspunkte, der die Kundensegmente beschreibt."
        "Bitte stellen Sie sicher, dass die Kundensegmente klar definiert und im Kontext dieser Rolle relevant sind."

        # Wertangebote
        "Wertangebote: Welchen einzigartigen Wert bringt diese Rolle den Kunden? Wie löst diese Rolle die Probleme der Kunden "
        "oder erfüllt ihre Bedürfnisse? Welche Vorteile oder Nutzen bietet sie? Wie differenziert sie sich von Wettbewerbern?\n\n"
        "Liefern Sie einen detaillierten Absatz, keine Aufzählungspunkte, der das Kernwertangebot beschreibt."
        "Erstellen Sie einen 300 Wörter langen Absatz, der alle Wertangebote hervorhebt."
        "Bitte stellen Sie sicher, dass die Wertangebote klar definiert und im Kontext dieser Rolle relevant sind."

        # Kundenbeziehungen
        "Kundenbeziehungen: Welche Art von Beziehung wird zu den Kunden aufgebaut und gepflegt? "
        "Umfasst diese Rolle persönliche Unterstützung, Selbstbedienung oder Automatisierung in der Kundenbeziehung? "
        "Wie stellt diese Rolle langfristige Kundenzufriedenheit sicher?\n\n"
        "Liefern Sie einen detaillierten Absatz, keine Aufzählungspunkte, der den Ansatz für Kundenbeziehungen erklärt."
        "Erstellen Sie einen 300 Wörter langen Absatz, der alle Kundenbeziehungen hervorhebt."
        "Bitte stellen Sie sicher, dass die Kundenbeziehungen klar definiert und im Kontext dieser Rolle relevant sind."

        # Kanäle
        "Kanäle: Welche primären Kanäle nutzt diese Rolle, um den Kunden ihren Wert zu liefern? Wie erreicht sie die Kunden, "
        "und welche Methoden oder Plattformen werden verwendet, um zu kommunizieren, zu verteilen oder Dienstleistungen anzubieten?\n\n"
        "Liefern Sie einen detaillierten Absatz, keine Aufzählungspunkte, der die Kommunikations- und Vertriebskanäle dieser Rolle beschreibt."
        "Erstellen Sie einen 300 Wörter langen Absatz, der alle Kanäle hervorhebt."
        "Bitte stellen Sie sicher, dass die Kanäle klar definiert und im Kontext dieser Rolle relevant sind."

        # Einnahmequellen
        "Einnahmequellen: Wie generiert diese Rolle Einnahmen? Welche verschiedenen Wege gibt es, um Geld zu verdienen? "
        "Stützt sie sich auf Direktverkäufe, Abonnements, Lizenzen oder andere Einnahmemethoden?\n\n"
        "Liefern Sie einen detaillierten Absatz, keine Aufzählungspunkte, der erklärt, wie diese Rolle zu den Einnahmequellen der Organisation beiträgt."
        "Erstellen Sie einen 300 Wörter langen Absatz, der alle Einnahmequellen hervorhebt."
        "Bitte stellen Sie sicher, dass die Einnahmequellen klar definiert und im Kontext dieser Rolle relevant sind."

        # Schlüsselressourcen
        "Schlüsselressourcen: Was sind die wesentlichen Ressourcen, die benötigt werden, damit diese Rolle ihr Wertangebot liefern kann? "
        "Dies könnte menschliche Ressourcen, physische Vermögenswerte, geistiges Eigentum oder finanzielle Ressourcen umfassen.\n\n"
        "Liefern Sie einen detaillierten Absatz, keine Aufzählungspunkte, der die notwendigen Schlüsselressourcen erklärt."
        "Erstellen Sie einen 300 Wörter langen Absatz, der alle Schlüsselressourcen hervorhebt."
        "Bitte stellen Sie sicher, dass die Schlüsselressourcen klar definiert und im Kontext dieser Rolle relevant sind."

        # Schlüsselaktivitäten
        "Schlüsselaktivitäten: Was sind die Kernaktivitäten, die diese Rolle durchführen muss, um ihr Wertangebot zu liefern? "
        "Dazu gehören die Hauptaufgaben und Verantwortlichkeiten, die für den Erfolg entscheidend sind.\n\n"
        "Liefern Sie einen detaillierten Absatz, keine Aufzählungspunkte, der die Schlüsselaktivitäten beschreibt."
        "Erstellen Sie einen 300 Wörter langen Absatz, der alle Schlüsselaktivitäten hervorhebt."
        "Bitte stellen Sie sicher, dass die Schlüsselaktivitäten klar definiert und im Kontext dieser Rolle relevant sind."

        # Schlüsselpartner
        "Schlüsselpartner: Wer sind die wichtigsten Partner oder Mitwirkenden, mit denen diese Rolle interagiert? "
        "Dazu können Lieferanten, Allianzen oder andere Stakeholder gehören, die für den Erfolg entscheidend sind.\n\n"
        "Liefern Sie einen detaillierten Absatz, keine Aufzählungspunkte, der die Schlüsselpartner beschreibt."
        "Erstellen Sie einen 300 Wörter langen Absatz, der alle Schlüsselpartner hervorhebt."
        "Bitte stellen Sie sicher, dass die Schlüsselpartner klar definiert und im Kontext dieser Rolle relevant sind."

        # Kostenstruktur
        "Kostenstruktur: Was sind die primären Kosten im Zusammenhang mit dieser Rolle? Dies könnte Gehälter, Betriebskosten, Ressourcen "
        "und andere Ausgaben umfassen. Wie stehen diese Kosten im Zusammenhang mit den Schlüsselressourcen, Aktivitäten und Partnerschaften?\n\n"
        "Liefern Sie einen detaillierten Absatz, keine Aufzählungspunkte, der die Kostenstruktur dieser Rolle erklärt."
        "Erstellen Sie einen 300 Wörter langen Absatz, der alle Kostenstrukturen hervorhebt."
        "Bitte stellen Sie sicher, dass die Kostenstruktur klar definiert und im Kontext dieser Rolle relevant ist."

        "Jeder Abschnitt muss ein 300 Wörter langer Absatz sein."
        "Keine Aufzählungspunkte in den Abschnitten, nur Absätze."
        "Bitte liefern Sie für jeden Abschnitt eine detaillierte Antwort."
        "Jeder Abschnitt ist ein Absatz!"
        )
        
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(
        prompt,
    )
    return response.text

import re

def extract_sections(response_text, language):
    sections = {}

    # Define the section markers for English and German
    section_titles_en = [
        "Customer Segments", "Value Proposition", "Customer Relationships",
        "Channels", "Revenue Streams", "Key Resources",
        "Key Activities", "Key Partners", "Cost Structure"
    ]
    section_titles_de = [
        "Kundensegmente", "Wertangebote", "Kundenbeziehungen",
        "Kanäle", "Einnahmequellen", "Schlüsselressourcen",
        "Schlüsselaktivitäten", "Schlüsselpartner", "Kostenstruktur"
    ]
    section_titles_fr = [
        "Segments de Clients", "Proposition de Valeur", "Relations Clients",
        "Canaux", "Sources de Revenus", "Ressources Clés",
        "Activités Clés", "Partenaires Clés", "Structure des Coûts"
    ]
    section_titles_es = [
        "Segmentos de Clientes", "Propuesta de Valor", "Relaciones con Clientes",
        "Canales", "Flujos de Ingresos", "Recursos Clave",
        "Actividades Clave", "Socios Clave", "Estructura de Costos"
    ]
    section_titles_it = [
        "Segmenti di Clienti", "Proposta di Valore", "Relazioni con i Clienti",
        "Canali", "Flussi di Entrate", "Risorse Chiave",
        "Attività Chiave", "Partner Chiave", "Struttura dei Costi"
    ]
    section_titles_nl = [
        "Klantsegmenten", "Waardepropositie", "Klantrelaties",
        "Kanalen", "Inkomstenstromen", "Key Resources",
        "Key Activities", "Key Partners", "Kostenstructuur"
    ]

    # Set the correct section titles based on the language
    if language.lower() == "en" or language.lower() == "english":
        section_titles = section_titles_en
    elif language.lower() == "de" or language.lower() == "german":
        section_titles = section_titles_de
    elif language.lower() == "fr" or language.lower() == "french":
        section_titles = section_titles_fr
    elif language.lower() == "es" or language.lower() == "spanish":
        section_titles = section_titles_es
    elif language.lower() == "it" or language.lower() == "italian":
        section_titles = section_titles_it
    elif language.lower() == "nl" or language.lower() == "dutch":
        section_titles = section_titles_nl
    else:
        raise ValueError("Unsupported language")

    # Helper function to find the closest section title using fuzzy matching
    def get_best_match(line, titles):
        max_score = 0
        best_match = None
        for title in titles:
            score = fuzz.partial_ratio(line.lower(), title.lower())  # Compare case-insensitively
            if score > max_score and score >= 80:  # A threshold to avoid poor matches
                max_score = score
                best_match = title
        return best_match
    
    # Start by splitting the response into lines
    lines = response_text.split("\n")

    current_section = None
    section_content = []

    # Go through each line and extract section content
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces
        
        # Skip empty lines
        if not line:
            continue
        # Try to find the best matching section title
        matched_title = get_best_match(line, section_titles)
        '''# Check if the line matches any section title
        matched_title = None
        for title in section_titles:
            if title.lower() in line.lower():  # Case-insensitive check
                matched_title = title
                break'''
        if matched_title:
            # If we have a current section, save its content
            if current_section:
                sections[current_section] = "\n".join(section_content).strip()

            # Remove the matched section title from the section titles list
            section_titles.remove(matched_title)
            
            # Start a new section
            current_section = matched_title
        else:
            # Add content to the current section
            if current_section:
                section_content.append(line)

    # Don't forget to save the last section
    if current_section:
        sections[current_section] = "\n".join(section_content).strip()

    return sections


# Define a response model (optional, but useful for clarity in your API)
class ProcessedDataResponse(BaseModel):
    message: str
    
class UserInputRequest(BaseModel):
    user_input: str
    language: str

# Create an endpoint to trigger the processing
@app.post("/process-data", response_model=ProcessedDataResponse)
def process_data(request: UserInputRequest):
    try:   
        # Extract the language from the request
        user_language = request.language
        #get the filepath of the pkl file we will use
        FilePath = get_file_path_by_language(user_language)
        grouped_df = pd.read_pickle(FilePath)
        # Finding n_matching occupations
        matched_occupations_str,matched_occupations_list = find_top_matching_occupations(FilePath,request.user_input,top_n=7)
        print("Matched occupations:", matched_occupations_str)
        #asking AI if occupation matches
        occupation_match, skills_paragraph = ask_AI(request.user_input,matched_occupations_str,user_language)
        print("user_language Match:", user_language)
        print("Occupation Match:", occupation_match)
        print("\nSkills Required:\n", skills_paragraph)
        
        #retireve a dict containing all matched occupations (because some occupations are friseur/friseurin and the ai will retrieve only friseur which is not an occupation)
        matched_occupations_dict=construct_dict_from_list(matched_occupations_list)
        if occupation_match != "no" and occupation_match != "" and occupation_match in matched_occupations_dict:
            occupation_match = matched_occupations_dict[occupation_match]
        print("Occupation Match NEWW:", occupation_match)
        
        #create the content that we will give in addition to our BMC prompt
        content =generate_content(request.user_input,occupation_match,skills_paragraph,get_all_occupation_informations,matched_occupations_list,user_language,grouped_df)
        print(content)
        BMC_response=process_full_BMC(content,user_language)
        print(BMC_response)
        # Extract sections
        bmc_sections = extract_sections(BMC_response,user_language)
        #JSONIFY the response
        bmc_sections_json = json.dumps(bmc_sections, indent=4, ensure_ascii=False)
        print(bmc_sections_json)
        return {"message": bmc_sections_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

