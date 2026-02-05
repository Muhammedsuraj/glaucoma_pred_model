import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
#from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import FunctionTransformer

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object, coerce_numeric_func

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ['age', 'gender', 'education', 'employment', 'weightkg1',
                'sys_press_meas1', 'dia_press_meas1', 'cigs_100', 'alcohol', 'hbp',
                'diab', 'famhist_glaucoma', 'vis_acuity_re_2', 'vis_acuity_le_2',
                'iop_re_1', 'iop_le_1', 'iop_re_high', 'iop_le_high', 'vis_field_md_re',
                'vis_field_psd_re', 'vis_field_ght_re', 'vis_field_md_le',
                'vis_field_psd_le', 'vis_field_ght_le', 'axial_re', 'axial_le',
                'thick_re', 'thick_le', 'sup_rnft_re', 'sup_rnft_le', 'inf_rnft_re',
                'inf_rnft_le', 'nasal_rnft_re', 'nasal_rnft_le', 'temp_rnft_re',
                'temp_rnft_le', 'mean_rnfl_thick_re', 'mean_rnfl_thick_le'
                ]
            
            categorical_columns = ['heightcm1']

            numeric_coercer = FunctionTransformer(
                coerce_numeric_func,
                feature_names_out="one-to-one"
            )


            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="mean")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("coerce_numeric", numeric_coercer),
                ("imputer",SimpleImputer(strategy="mean")),
                #("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
                #("target_encoder",TargetEncoder(smoothing=10, min_samples_leaf=5))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="casetype"
            numerical_columns = ['age', 'gender', 'education', 'employment', 'weightkg1',
                'sys_press_meas1', 'dia_press_meas1', 'cigs_100', 'alcohol', 'hbp',
                'diab', 'famhist_glaucoma', 'vis_acuity_re_2', 'vis_acuity_le_2',
                'iop_re_1', 'iop_le_1', 'iop_re_high', 'iop_le_high', 'vis_field_md_re',
                'vis_field_psd_re', 'vis_field_ght_re', 'vis_field_md_le',
                'vis_field_psd_le', 'vis_field_ght_le', 'axial_re', 'axial_le',
                'thick_re', 'thick_le', 'sup_rnft_re', 'sup_rnft_le', 'inf_rnft_re',
                'inf_rnft_le', 'nasal_rnft_re', 'nasal_rnft_le', 'temp_rnft_re',
                'temp_rnft_le', 'mean_rnfl_thick_re', 'mean_rnfl_thick_le'
                ]

            input_feature_train_df=train_df.drop(target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
