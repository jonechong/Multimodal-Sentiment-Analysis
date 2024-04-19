import streamlit as st

st.title('Project Overview')
st.header('Project Pipeline')

#overview
class project_pipeline():
    st.image('../assets/CDS_pipeline.png')
    st.write('Our project pipeline focuses on extracting **3 main features** from our video input, and perform feature manipulation to finally get a **sentiment output** of that video.')
    st.divider()
class dataset():
    st.header('Dataset')
    st.write('The three dataset we looked at are MOSI, MOSEI and IEMOCAP which are in the format of .mp4 and pickle')
    st.subheader('Dataset visualization')
    st.image('../assets/dataset1.png', caption='The video lengths distribution shows that most of the videos in our dataset fall below 10-20 seconds')
    st.image('../assets/dataset2.png', caption='The uneven distribution shown prompts us to do data manipulation during our dataloading.')
    st.subheader("Rationale")
    st.write('We chose this dataset as it is open source and has a rich multimodal data. This dataset is also widely used in the research community and is rigorously annotated for sentiment analysis.')
    st.divider()
class feature_processing():
    st.header('Feature Processing')
    st.subheader('Text Processing')
    st.image('../assets/text_extraction_pipeline.png', caption='Using Mozilla DeepSpeech, BERT and Autoencoder.')
    st.subheader('Text Preprocessing Visualization')
    st.image('../assets/text1.png')
    st.image('../assets/text2.png')
    st.write("After dimensionality reduction, **no specific observable pattern** (as shown). One reason is that the dimensionality reduction resulted in **loss of significant data**. Hence, the distinctiveness of each sentiment with respect to their BERT features may not be visualized.")

    st.subheader('Audio Processing')
    st.markdown(
    """
    For audio extraction, we used:
    - Mel-Frequency Cepstral Coefficients (MFCCs)
    - Chroma Feature
    - Spectral Contrast
    - Tonnetz (Tonal Centroid Features)
    """
    )
    st.subheader('Audio Preprocessing Visualization')
    st.image('../assets/audio1.png')
    st.image('../assets/audio2.png', caption="The correlation heatmap shows that the features are more or less independent with respect to each other, except some.")

    st.subheader('Facial Processing')
    st.image('../assets/facial_extraction_pipeline.png', caption='Using OpenCV Haar Cascade and ResNet50')
    st.subheader('Facial Preprocessing Visualization')
    st.image('../assets/facial1.png')
    st.image('../assets/facial2.png')

    st.divider()

class performance():
    st.header('Model Performance and Evaluation Matrices')
    st.write('Insert some evaluation and performance here')

st.divider()

class future_improvements():
    st.header("Future improvements")
    st.write('Insert some idea for future improvements or alternative solutions')


