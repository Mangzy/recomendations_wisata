from flask import Flask, Response, request
from flasgger import Swagger 
from sklearn.metrics.pairwise import linear_kernel
import joblib
import bnlearn as bn
import json

app = Flask(__name__)
swagger_template = {
    'info':{
        'title': 'API Recomendation Wisata Surabaya',
        'description': 'Test',
        'versions' : '1.0.0'
    }
}
swagger_config = {
    'headers': [
    ],
    'specs': [
        {
            'endpoint': 'apispec_1',
            'route': '/apispec_1.json',
            'rule_filter': lambda rule: True,
            'model_filter': lambda tag: True,
        }
    ],
    'static_url_path': '/flasgger_static',
    'swagger_ui': True,
    'specs_route': '/apidocs/'
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)

@app.route('/')
def index():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    # Load Model
    model = joblib.load('model.pkl')
    data = request.get_json()
    escape = data.get('Escape')
    relaxation = data.get('Relaxation')
    play = data.get('Play')
    strengthen = data.get('Strenghthening family bonds')
    prestige = data.get('Prestige')
    social = data.get('Social Interaction')
    romance = data.get('Romance')
    educational = data.get('Educational Opportunity')
    self = data.get('Self-fulfilment')
    wish = data.get('Wish-fulfiment')
    lingkungan = data.get('Lingkungan')
    infrastruktur = data.get('Infrastruktur')
    fasilitas = data.get('Fasilitas')
    akomodasi = data.get('Akomodasi')
    makan = data.get('Makan Bersama')
    berolahraga = data.get('Berolahraga')
    belajar = data.get('Belajar')
    berinteraksi = data.get('Berinteraksi dengan satwa')
    foto = data.get('Mengambil Foto')
    beribadah = data.get('Beribadah')
    berkemah = data.get('Berkemah')
    pemandangan = data.get('Melihat Pemandangan')
    berbelanja = data.get('Berbelanja')
    evidence = {
        'Escape': escape,
        'Relaxation': relaxation,
        'Play': play,
        'Strenghthening family bonds': strengthen,
        'Prestige': prestige,
        'Social Interaction': social,
        'Romance': romance,
        'Educational Opportunity': educational,
        'Self-fulfilment': self,
        'Wish-fulfiment': wish,
        'Lingkungan': lingkungan,
        'Infrastruktur': infrastruktur,
        'Fasilitas': fasilitas,
        'Akomodasi': akomodasi,
        'Makan Bersama': makan,
        'Berolahraga': berolahraga,
        'Belajar': belajar,
        'Berinteraksi dengan satwa': berinteraksi,
        'Mengambil Foto': foto,
        'Beribadah': beribadah,
        'Berkemah': berkemah,
        'Melihat Pemandangan': pemandangan,
        'Berbelanja': berbelanja
    }
    query = bn.inference.fit(model, variables=['Nama Wisata'], evidence=evidence, verbose=3)
    query.df['Nama Wisata'] = {
        0: "Balai Pemuda Alun - Alun Surabaya",
        1: "Galaxy Mall Surabaya",
        2: "House of Sampoerna",
        3: "Jalan Tunjungan",
        4: "Jatim International Expo (JIE) Convention Exhibiton",
        5: "Kampung Lawas Maspati",
        6: "Kampung Pecinaan Kapasan Dalam",
        7: "Kebun Binatang Surabaya",
        8: "Kebun Raya Mangrove Gunung Anyar",
        9: "Kodam Street Food Surabaya",
        10: "Kuliner Pecinan Kembang Jepun",
        11: "Makam Sunan Ampel",
        12: "Masjid Cheng Ho",
        13: "Masjid Nasional Al Akbar",
        14: "Mirota Batik & Handicraft",
        15: "Monumen Tugu Pahlawan",
        16: "Monumen Kapal Selam",
        17: "Museum Pendidikan Surabaya",
        18: "Museum Surabaya Gedung Siola",
        19: "Pasar Genteng",
        20: "Pasar Pabean",
        21: "Pusat Olahraga KONI",
        22: "Stadion Gelora Bung Tomo",
        23: "Surabaya Convention Center",
        24: "Taman Bungkul",
        25: "Taman Suroboyo",
        26: "Tunjungan Plaza"
    }
    query.df = query.df.sort_values(by='p', ascending=False)
    
    wisata_list = []
    print("Rekomendasi Wisata:")
    # ambil 5 nilai tertinggi dan taruh ke dalam wisata_list
    for index, row in query.df.head(5).iterrows():
        nama_wisata = row['Nama Wisata']
        wisata_list.append({'Nama Wisata': nama_wisata})
    recipe_json = json.dumps(wisata_list)
    response = Response(
        response=recipe_json,
        status=200,
        mimetype='application/json'
    )

    return response  
