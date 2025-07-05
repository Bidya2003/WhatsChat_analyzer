# ner.py

import spacy
from collections import defaultdict, Counter
from geopy.geocoders import Nominatim
import folium

nlp = spacy.load("en_core_web_sm")

def extract_named_entities(messages):
    grouped_entities = defaultdict(list)
    for text in messages:
        doc = nlp(str(text))
        for ent in doc.ents:
            grouped_entities[ent.label_].append(ent.text)
    return grouped_entities

def get_top_entities(entities, entity_type="PERSON", top_n=10):
    return Counter(entities.get(entity_type, [])).most_common(top_n)



def generate_location_map(locations, filename="ner_map.html"):
    geolocator = Nominatim(user_agent="whatsapp-ner")
    m = folium.Map(location=[20, 0], zoom_start=2)

    for loc in set(locations[:50]):
        try:
            location = geolocator.geocode(loc)
            if location:
                folium.Marker([location.latitude, location.longitude], tooltip=loc).add_to(m)
        except:
            continue

    m.save(filename)
    return filename


