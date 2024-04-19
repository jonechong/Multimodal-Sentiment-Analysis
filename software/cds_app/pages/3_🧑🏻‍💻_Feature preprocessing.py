import streamlit as st

st.title('Preprocessing')

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


