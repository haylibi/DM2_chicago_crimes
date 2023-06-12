import pandas as pd
import re
import nltk
import spacy as sp
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


def reduce_location_description(description):

    # Reduce or categorize the location descriptions based on predefined keyword mappings. 
    #Its purpose is to simplify and standardize the descriptions by assigning them to specific categories or labels.
    
    #These clusters come from (2)clustering
    #But before using it here there were some changes, some words were replaced, and some clusters were made manually
    """
    #LOCATION
    keyword_mapping = {
        'commercial': ['manufacturing', 'dealership', 'commercial', 'business', 'factory', 'retail', 'warehouse'],
        'transportation': ['transportation', 'railroad', 'subway', 'train', 'bus', 'uber', 'lyft'],
        'college': ['university', 'grammar', 'college', 'school'],
        'street': [ 'street', 'track', 'horse', 'alley', 'ride'],
        'care': ['hospital', 'nursing', 'medical', 'dental', 'care'],
        'building': ['establishment', 'construction', 'abandoned', 'platform', 'building', 'ground', 'lobby', 'small', 'site', 'area'],
        'entertainment': ['entertainment', 'theater', 'movie', 'bowling',],
        'police': ['police', 'jail', 'fire'],
        'store': ['convenience', 'newsstand', 'grocery', 'vending', 'liquor', 'store', 'shop'],
        'residential': ['residential', 'waterfront', 'lakefront', 'residence', 'apartment', 'property', 'vacant', 'land', 'sidewalk'],
        'highway': ['expressway', 'highway'],
        'department': ['department', 'government', 'retirement', 'private', 'federal', 'service', 'public', 'office', 'system', 'party', 'union'],
        'watercraft': ['watercraft', 'aircraft', 'boat'],
        'restaurant': ['restaurant', 'tavern', 'hotel', 'motel', 'bar', 'food'],
        'beauty': ['barbershop', 'beauty', 'barber', 'salon'],
        'church': ['synagogue', 'worship', 'church'],
        'driveway': ['riverbank', 'elevator', 'exterior', 'basement', 'driveway', 'parking', 'garage', 'porch'],
        'operated': ['terminal', 'facility', 'airport', 'depot', 'yard'],
        'hallway': ['stairwell', 'mezzanine', 'vestibule', 'gangway', 'hallway'],
        'machine': ['automatic', 'machine', 'teller', 'atm'],
        'gas': ['delivery', 'specify', 'gas'],
        'cemetary': ['cemetary'],
        'bank': ['currency', 'exchange', 'share', 'credit', 'bank', 'loan'],
        'repair': ['cleaning', 'repair', 'wash'],
        'secure': ['preserve', 'stable', 'secure', 'saving'],
        'sport': ['athletic', 'sport', 'club', 'stadium', 'arena'],
        'taxicab': ['taxicab', 'rooming', 'trolley', 'cta'],
        'center': ['banquet', 'library', 'center', 'ymca', 'hall', 'room'],
        'vehicle': ['vehicle', 'trailer', 'truck', 'car'],
        'bridge': ['bridge', 'river'],
        'animal': ['animal', 'farm'],
        'forest': ['forest', 'park', 'pool', 'lake', 'wooded']
    }
    

    """
    #DESCRIPTION
    keyword_mapping = {
        'synthetic': ['synthetic', 'tar','black', 'brown', 'white','tan'],
        'drug': ['methamphetamine', 'prescription', 'amphetamine', 'marijuana', 'cannabis', 'narcotic', 'cocaine', 'heroin', 'drug', 'gram','crack', 'pcp'],
        'hallucinogen': ['paraphernalia', 'hallucinogen', 'intoxicating', 'compounding', 'liquor','barbiturate'],
        'abuse': ['abuse', 'child'],
        'sexual': ['sexual', 'sex', 'violation'],
        'bullet': ['bullet', 'knife', 'foot', 'arm'],
        'prostitution' : ['prostitution', 'prostitute', 'pimping'],
        'stalking': ['impersonation', 'cyberstalking',  'patronizing',  'pandering', 'stalking', 'peeping', 'bigamy'],
        'videotaping': ['eavesdropping', 'videotaping', 'endangering', 'obstructing', 'tampering', 'obstruct', 'tamper', 'aiding'],
        'extortion': ['embezzlement', 'laundering', 'conspiracy', 'extortion', 'bribery', 'forgery', 'fraud'],
        'obscene': ['pornographic', 'pornography', 'obscenity', 'indecency', 'indecent', 'obscene','fornication'],
        'neglect': ['abandonment', 'mutilation', 'servitude', 'forcible', 'neglect', 'unborn'],
        'violent': ['demonstration', 'involving', 'fighting', 'invasion', 'violence', 'incident', 'violent', 'attempt', 'serious', 'police', 'threat', 'action', 'escape', 'armed', 'fight', 'fire'],
        'vehicle': ['automobile', 'vehicle', 'scooter', 'truck', 'motor', 'bike', 'bus', 'cab'],
        'money': ['receive', 'credit', 'money', 'card', 'sale', 'sell', 'cash', 'pay', 'buy'],
        'weapon': ['firearm', 'handgun', 'weapon', 'rifle', 'gun'],
        'vandalism': ['vandalism', 'burglary', 'looting', 'stolen', 'arson', 'theft'],
        'murder': ['kidnapping', 'abduction', 'hijacking', 'criminal', 'homicide', 'suspect', 'victim', 'murder', 'crime'],
        'identification': ['identification', 'registration', 'licensed', 'register', 'identity', 'passport', 'license', 'entry'],
        'counterfeit': ['counterfeiting', 'counterfeit'],
        'unidentifiable': ['unidentifiable',  'firework', 'arsonist', 'mislaid'],
        'electronic': ['instrument', 'manufacture', 'electronic', 'equipment', 'computer', 'machine', 'metal', 'tool'],
        'exploit': ['exploitation', 'exploit', 'labor'],
        'defacement': ['misrepresent', 'defacement', 'patronize', 'advertise', 'deface'],
        'intimidation': ['intimidation', 'harassment', 'misconduct'],
        'interference': ['interference', 'interfere', 'influence', 'restraint', 'resist', 'arrestee'],
        'transmission': ['transmission', 'hiv'],
        'notification': ['notification', 'revocation', 'visitation', 'forfeit', 'lessee'],
        'concealed': ['unauthorized', 'possession', 'prohibited', 'concealed', 'substance', 'unlawful', 'violate', 'illegal', 'intent'],
        'commercial': ['consumption', 'commercial', 'financial', 'business', 'domestic', 'product', 'retail'],
        'delinquency': ['delinquency', 'juvenile', 'adult'],
        'school': ['educational', 'education', 'degree', 'school'],
        'judicial': ['enforcement', 'protection', 'protected', 'judicial', 'justice', 'civil', 'act', 'law'],
        'conduct': ['contribute', 'distribute', 'disclose', 'refusing', 'practice', 'conduct', 'deliver', 'disarm', 'employ', 'carry', 'fail', 'use'],
        'injury': ['offense', 'injury', 'minor', 'game'],
        'collection': ['collection', 'library', 'print', 'image'],
        'sound': ['controlled', 'sound', 'alarm', 'zone'],
        'needle': ['hypodermic', 'needle'],
        'armor': ['ammunition', 'piercing', 'armor', 'ammo'],
        'explosive': ['incendiary', 'explosive', 'bomb'],
        'confession': ['confession', 'bogus', 'false'],
        'recording': ['recording', 'recorded', 'record'],
        'predatory': ['predatory', 'deceptive'],
        'elderly': ['insurance', 'disabled', 'elderly', 'health', 'care'],
        'probation': ['probation', 'detention', 'parole', 'bail'],
        'passageway': ['amusement', 'passageway', 'tavern', 'forge'],
        'duty': ['discharge', 'service', 'battery', 'maintenance', 'duty'],
        'location': ['container', 'operation', 'facility', 'location', 'operated', 'closure', 'harbor', 'plant'],
        'emergency': ['emergency', 'delivery', 'cutting', 'aid', 'cut'],
        'runaway': ['snatching', 'stranger', 'runaway', 'picking', 'alike', 'posse'],
        'residence': ['residence', 'compound', 'building', 'family', 'home', 'motorhome'],
        'contact': ['telephone', 'contact', 'address', 'call'],
        'citizen': ['official', 'employee', 'citizen', 'officer', 'member', 'senior', 'board'],
        'property': ['property', 'private', 'public', 'state', 'land'],
        'tobacco': ['tobacco', 'smoking'],
        'airport': ['airport', 'plane', 'air'],
        'animal': ['animal', 'found', 'cycle', 'body'],
        'coin': ['pocket', 'purse', 'coin'],
        'mobile': ['mobile', 'device', 'gps', 'app'],
        'dangerous': ['dangerous', 'hazardous'],
        'aggravated': ['manslaughter', 'involuntary', 'aggravated', 'vehicular', 'reckless', 'assault'],
        'institutional': ['institutional', 'confidence', 'supported', 'recovery', 'bond'],
        'information': ['dissemination', 'information', 'monitoring', 'document', 'material', 'process', 'summary', 'report'],
    }
    
    
    nlp = sp.load('en_core_web_sm')
    description_doc = nlp(description)
    
    tokens = nltk.word_tokenize(description.lower())
    
    #checks the similarity between the description_doc and each keyword using spaCy's similarity method
    #if the similarity score is above or equal to 0.9, it returns the corresponding key
    for key, keywords in keyword_mapping.items():
        for keyword in keywords:
            keyword_doc = nlp(keyword)
            similarity = description_doc.similarity(keyword_doc)
            if similarity >= 0.9:
                return key
            
    #It calculates the partial ratio between the description and keyword, and if it is above or equal to 90, it returns the corresponding key
    for key, keywords in keyword_mapping.items():
        for keyword in keywords:
            if fuzz.partial_ratio(description, keyword) >= 90:
                return key
    
    #checks if any word from word_list is present in the tokens list (which contains the tokenized description)
    # If any word is found, it returns the corresponding key
    for key, word_list in keyword_mapping.items():
        if any(word in tokens for word in word_list):
            return key
    
    #The first iteration uses spaCy's similarity score, the second one employs fuzzy string matching, and the third  one checks for direct word matches
    return description

# Read location or description from CSV
#these csv were generated by selecting unique location and description
#df = pd.read_csv('location.csv')
df = pd.read_csv('description.csv')
descriptions = df['Description'].tolist()

# Preprocess and reduce location descriptions
preprocessed_descriptions = [preprocess_text(desc) for desc in descriptions]
reduced_descriptions = [reduce_location_description(desc) for desc in preprocessed_descriptions]

#this is the final result, it is used to build the graph for link analysis
# Define the output file path
output_file = 'reduced_result_description.txt'
#output_file = 'reduced_result_location.txt'

# Save the reduced descriptions to the text file
with open(output_file, 'w') as file:
    for desc in reduced_descriptions:
        file.write(desc + '\n')
