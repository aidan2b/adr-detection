import requests
import json
import pandas as pd

def get_faers(medication):
    
    print(f"Fetching FAERS data for: {medication}")
    
    print(f"Medication variable type: {type(medication)}")

    accepted = False
    try:
        response = requests.get('https://api.fda.gov/drug/event.json?search=patient.drug.openfda.brand_name:'+medication+'&limit=20&count=patient.reaction.reactionmeddrapt.exact')
        data = response.json()
        df = pd.DataFrame(data['results'])
        accepted = True
    except:
        accepted = False

    if accepted == False:
        try:
            response = requests.get('https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:'+medication+'&limit=20&count=patient.reaction.reactionmeddrapt.exact')
            data = response.json()
            df = pd.DataFrame(data['results'])
            accepted = True
        except:
            accepted = False
    if accepted == False:
        print(f"{medication} invalid")
    else:
        df.to_csv('faers.csv')