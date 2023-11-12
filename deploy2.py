import dash
from dash import Dash, Input, Output, State, dcc, html, callback
from flask import Flask, send_file
import pandas as pd
import pickle
import numpy as np
import os
import base64  # Import the base64 module
import io

# Initialize your Dash app
app = dash.Dash(__name__)

# Create server variable with Flask server object for use with gunicorn
server = app.server # Flask server

# Define the app layout

app.layout = html.Div([
    html.H1("Theme: Food Security and Access- to prevent malnutrition"),
    html.H2(" "),
    html.H2("Topic: Food accessibility and Profitability"),
    html.H2(" "),
    html.H2("Group Names: ML-Explorer"),
    html.H2(" "),
    dcc.Markdown('''
        ## A prediction app to accurately assess the factors contributing to food accessibility and profitability

        For farming households in Africa, by identifying the crucial determinants affecting food accessibility and profitability, thereby providing insights that can inform policymakers, non-governmental organisations, and other stakeholders in the agricultural sector. Through this prediction app, we seek to contribute to the advancement of sustainable agricultural practices and the improvement of food security in the region.
    '''),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
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
    html.Button("Download File", id="btn-download"),
])

# Load your trained model and other necessary data (e.g., label encoders) here
input_file = "model_gmb_1.pkl"
input_file2 = "le_1.pkl"

with open(input_file, 'rb') as f_in: 
    model = pickle.load(f_in)
    
with open(input_file2, 'rb') as f_in: 

    le = pickle.load(f_in)

# Define a callback to handle the file upload and perform prediction
@app.callback(Output('output-prediction', 'children'), [Input('upload-data', 'contents')])
def predict_file(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        uploaded_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        df = uploaded_data
        
        # Perform any necessary data preprocessing (e.g., using label encoders)
        df.dropna(subset=["s1quant_sold"], inplace=True)
        uploaded_data.dropna(subset=["s1quant_sold"], inplace=True)
        del df["hhcode"]
        
        num_cols = df.select_dtypes("number").columns
        cat_cols = df.select_dtypes("object").columns
        
        # Fill missing values
        for col in num_cols:
            df[col].fillna(0, inplace=True)

        for c in cat_cols:
            nam = 'Unknown_%s' % c
            df[c].fillna(nam, inplace=True)

        for c in df.select_dtypes("object").columns:
            # perform label encoding Dataset
            df[c] = le.fit_transform(df[c])
            
        del df["s1quant_sold"]
        feat = ['Country', 'Region', 'fsystem1', 'tenure1', 'yearsuse1', 'rentplot1', 's1start', 's1end', 
                'seas1nam', 's1plant_data', 's1land_area', 's1quant_harv', 's1consumed', 's1livestock',
                's1lost', 's1market', 's1crop_val', 's1no_seed', 'pc1', 'nyieldc1', 's1irrig1',
                's1irrig2', 's1irrig3', 's1irrig4', 's1pest', 's1wat1', 's1wat2', 's1wat3', 's1wat4', 's1wat5',
                'costkgfert', 'costkgpest', 'distsmktkm', 'distsmkthr', 'distpmktkm', 'distpmkthr', 'transport',
                'cost1crop', 'cost2crop', 'cost3crop', 'cost5crop', 'farmingexperience', 'ad711', 'ad718', 'ad7111',
                'ad7116', 'ad7120', 'ad732', 'ad742', 'ad7511', 'ad7610', 'ad7613', 'ad7624']
        df = df[feat]
        
        # Make predictions using the loaded model
        predictions = model.predict(df)
        result = np.expm1(predictions).round(2)
        uploaded_data["Predictions"] = result
        
        # saving the dataframe
        csv_file_path = "Hamoye_Pred.csv"
        uploaded_data.to_csv(csv_file_path, index=False)
        
        return f"Predictions saved to {csv_file_path}"

    else:
        return 'Upload a file to make predictions'

# Callback to handle the file download
@app.callback(
    Output('btn-download', 'n_clicks'),
    Input('btn-download', 'n_clicks'),
    prevent_initial_call=True  # This ensures that the callback doesn't run on app startup
)
def download_file(n_clicks):
    '''Function to handle download'''
    if n_clicks is None:
        return

    # Serve the file using Flask's send_file
    return send_file(
        "Hamoye_Pred.csv",  # Use the correct file path
        mimetype='text/csv',  # Set the correct MIME type for CSV
        as_attachment=True,
        download_name='Hamoye_Pred.csv'
    )

if __name__ == '__main__':
    app.run_server(debug=True, jupyter_mode="tab", port='8996')
