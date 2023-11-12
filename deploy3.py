from dash import Dash, Input, Output, State, dcc, html, callback
from flask import send_file
import pandas as pd
import pickle
import numpy as np
import os
import base64  # Import the base64 module
import io
from urllib.parse import quote as urlquote

# Initialize your Dash app
app = Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Theme: Food Security and Access- to prevent malnutrition"),
    html.H2("Topic: Food accessibility and Profitability"),
    html.H2("Group Names: ML-Explorer"),
    dcc.Markdown('''
        ## A prediction app to accurately assess the factors contributing to food accessibility and profitability

        For farming households in Africa, by identifying the crucial determinants affecting food accessibility and profitability, thereby providing insights that can inform policymakers, non-governmental organisations, and other stakeholders in the agricultural sector. Through this prediction app, we seek to contribute to the advancement of sustainable agricultural practices and the improvement of food security in the region.
    '''),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False  # Set to True if you want to allow multiple files
    ),
    html.Div(id='output-prediction'),
])

# Load your trained model and other necessary data (e.g., label encoders) here
input_file_model = "model_gmb_1.pkl"
input_file_le = "le_1.pkl"

with open(input_file_model, 'rb') as f_in:
    model = pickle.load(f_in)

with open(input_file_le, 'rb') as f_in:
    le = pickle.load(f_in)


# Define a callback to handle the file upload and perform prediction
@app.callback(Output('output-prediction', 'children'), [Input('upload-data', 'contents')])
def predict_file(contents):
    if contents is not None:
        # Process uploaded file
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        uploaded_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Perform data preprocessing
        uploaded_data = preprocess_data(uploaded_data, le)

        # Make predictions using the loaded model
        predictions = model.predict(uploaded_data)
        result = np.expm1(predictions).round(2)
        uploaded_data["Predictions"] = result
        
        # Save the dataframe
        csv_file_path = "Hamoye_Pred.csv"
        uploaded_data.to_csv(csv_file_path, index=False)
        
        # Display the predictions as HTML or in a table
        return html.Div([html.H4(file_download_link(csv_file_path))])

    else:
        return 'Upload a file to make predictions'


def preprocess_data(data, label_encoder):
    # Perform any necessary data preprocessing (e.g., using label encoders)
    data.dropna(subset=["s1quant_sold"], inplace=True)
    data.drop(columns=["hhcode"], inplace=True)

    num_cols = data.select_dtypes("number").columns
    cat_cols = data.select_dtypes("object").columns
    
    # Fill missing values
    data[num_cols] = data[num_cols].fillna(0)
    data[cat_cols] = data[cat_cols].fillna('Unknown_' + data[cat_cols].astype(str))
    
    # Perform label encoding
    data[cat_cols] = data[cat_cols].apply(lambda col: label_encoder.fit_transform(col))
    
    # Select relevant features
    selected_features = ['Country', 'Region', 'fsystem1', 'tenure1', 'yearsuse1', 'rentplot1', 's1start', 's1end', 
                         'seas1nam', 's1plant_data', 's1land_area', 's1quant_harv', 's1consumed', 's1livestock',
                         's1lost', 's1market', 's1crop_val', 's1no_seed', 'pc1', 'nyieldc1', 's1irrig1',
                         's1irrig2', 's1irrig3', 's1irrig4', 's1pest', 's1wat1', 's1wat2', 's1wat3', 's1wat4', 's1wat5',
                         'costkgfert', 'costkgpest', 'distsmktkm', 'distsmkthr', 'distpmktkm', 'distpmkthr', 'transport',
                         'cost1crop', 'cost2crop', 'cost3crop', 'cost5crop', 'farmingexperience', 'ad711', 'ad718', 'ad7111',
                         'ad7116', 'ad7120', 'ad732', 'ad742', 'ad7511', 'ad7610', 'ad7613', 'ad7624']
    data = data[selected_features]
    
    return data


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


@app.server.route("/download/<path:path>")
def download(path):
    return send_file(
        path,  # Use the correct file path
        mimetype='text/csv',  # Set the correct MIME type for CSV
        as_attachment=True,
        download_name='Hamoye_Pred.csv'
    )

if __name__ == '__main__':
    app.run(debug=True, jupyter_mode="tab", port='8896')
