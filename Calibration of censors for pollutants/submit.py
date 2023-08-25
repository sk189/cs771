import pandas as pd
import pickle as pkl

def my_predict( df ):
	with open( "my_model", "rb" ) as file:
		model_o3 = pkl.load( file )
		model_no2 = pkl.load( file )
	
	df['Time'] = pd.to_datetime(df['Time'])
	df['hour'] = df['Time'].dt.hour
	df['minute'] = df['Time'].dt.minute

	X = df[['hour','minute','temp', 'humidity', 'no2op1', 'no2op2', 'o3op1', 'o3op2']]

	pred_o3 = model_o3.predict(X)
	pred_no2 = model_no2.predict(X)

	return ( pred_o3, pred_no2 )
