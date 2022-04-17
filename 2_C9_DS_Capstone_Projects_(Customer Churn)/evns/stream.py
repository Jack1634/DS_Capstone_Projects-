from IPython.core.display import HTML
import streamlit as st
import pandas as pd
import base64
import joblib

st.set_page_config(
    page_title='Employee Decision Predictor',
    page_icon='icon.png'
)

st.markdown(
    """
<h2 style='text-align:center; margin-bottom: -35px; color:#d33685;'>
WILL YOUR EMPLOYEE LEAVE OR STAY ??? </h2><br>""",unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1.5,1])
with col1:
    st.write("")
with col2:
    st.image("stay_or_go.webp", width=300)
with col3:
    st.write("")

st.markdown("""<h6 style='text-align:center;'>As a company owner, do you want to learn whether your employees will stay with you or leave?</h6>
<h6 style='text-align:center;'>Let's find out !!!</h6>
""", unsafe_allow_html=True)

def user_input_features():
    satisfaction_level = st.sidebar.slider("Satisfaction Level", 0.09, 1.0, 0.5)
    last_evaluation = st.sidebar.slider("Evaluation Score", 0.360, 1.0, 0.5)
    number_project = st.sidebar.selectbox('Number of Projects', [2, 3, 4, 5, 6, 7])
    average_montly_hours = st.sidebar.number_input('Avg. Monthly Working Hours', min_value=96, max_value=310, value=200, step=1)
    time_spend_company = st.sidebar.selectbox('Years in the Company', [2, 3, 4, 5, 6, 7, 8, 10])
    agree1 = st.sidebar.checkbox('Employee had a work accident')
    if agree1:
        Work_accident = 1
    else:
        Work_accident = 0
    agree2 = st.sidebar.checkbox('The employee received a promotion in the last 5 years')
    if agree2:
        promotion_last_5years = 1
    else:
        promotion_last_5years = 0
    choice = st.sidebar.selectbox('Department',
        ("Information Technology (IT)",
        "Research and Development (R & D)",
        "Accounting",
        "Human Resources",
        "Management",
        "Marketing",
        "Product Management",
        "Sales",
        "Support",
        "Technical"),
    )
    if choice == "Information Technology (IT)":
        departments = "IT"
    elif choice == "Research and Development (R & D)":
        departments = "RandD"
    elif choice == "Accounting":
        departments = "accounting"
    elif choice == "Human Resources":
        departments = "hr"
    elif choice == "Management":
        departments = "management"
    elif choice == "Marketing":
        departments = "marketing"
    elif choice == "Product Management":
        departments = "product_mng"
    elif choice == "Sales":
        departments = "sales"
    elif choice == "Support":
        departments = "support"
    elif choice == "Technical":
        departments = "technical" 
    choice2 = st.sidebar.radio('Salary Level', ["Low", "Medium", "High"])
    if choice2 == "Low":
        salary = "low"
    elif choice2 == "Medium":
        salary = "medium"
    elif choice2 == "High":
        salary = "high"
    new_df = {"satisfaction_level":satisfaction_level,
              "last_evaluation":last_evaluation,
              "number_project":number_project,
              "average_montly_hours":average_montly_hours,
              "time_spend_company":time_spend_company,
              "Work_accident":Work_accident,
              "promotion_last_5years":promotion_last_5years,
              "departments":departments,
              "salary":salary}
    features = pd.DataFrame(new_df, index=[0])
    return features
input_df = user_input_features()

st.markdown("""<h4 style='text-align:left; color:#d33685;'>Employee Features</h4>
""", unsafe_allow_html=True
)
resa = input_df.rename(columns={"satisfaction_level":"Satisfaction Level",
              "last_evaluation":"Evaluation Score",
              "number_project":"# of Projects",
              "average_montly_hours":"Monthly Hours",
              "time_spend_company":"Years in Company",
              "Work_accident":"Work Accident",
              "promotion_last_5years":"Received Promotion",
              "departments":"Department",
              "salary":"Salary Level"})
resa['Monthly Hours'] = resa['Monthly Hours'].astype('int')
resa['Work Accident'] = resa['Work Accident'].map({0:'No', 1:'Yes'})
resa['Received Promotion'] = resa['Received Promotion'].map({0:'No', 1:'Yes'})
resa['Department'] = resa['Department'].map({'IT':'Information Technology',
                        'RandD': 'R & D',
                        'accounting':'Accounting',
                        'hr':'Human Resources',
                        'management':'Management',
                        'marketing':'Marketing',
                        'product_mng':'Product Management',
                        'sales':'Sales',
                        'support':'Support',
                        'technical':'Technical'})
resa['Salary Level'] = resa['Salary Level'].map({'low':'Low', 'medium':'Medium', 'high':'High'})

st.write(HTML(resa.to_html(index=False, justify='left')))

gbc_model = joblib.load(open('gbc_model.pkl', 'rb'))
knn_model = joblib.load(open('knn_model.pkl', 'rb'))
rf_model = joblib.load(open('rf_model.pkl', 'rb'))

pred_gbc = gbc_model.predict(input_df)
pred_gbc = ['Left' if pred_gbc == 1 else 'Stayed']
pred_knn = knn_model.predict(input_df)
pred_knn = ['Left' if pred_knn == 1 else 'Stayed']
pred_rf = rf_model.predict(input_df)
pred_rf = ['Left' if pred_rf == 1 else 'Stayed']

st.title('')

# 1. Model (3 model yan yana)
# if st.button("Predict"):

#     st.subheader('Predictions')
#     col1, col2, col3 = st.columns(3)
#     col1.metric('Gradient Boosting', pred_gbc[0])
#     col2.metric('K-Nearest Neighbors', pred_knn[0])
#     col3.metric('Random Forest', pred_rf[0])

# 2. Model (model secimli)
st.markdown("""<h4 style='text-align:left; color:#d33685;'>Choose Your Model</h4>
""", unsafe_allow_html=True)

model = st.selectbox('Pick one model and get your prediction',
    ['Gradient Boosting', 'K-Nearest Neighbors', 'Random Forest'])

if model == 'Gradient Boosting':
    x1 = pred_gbc
elif model == 'K-Nearest Neighbors':
    x1 = pred_knn
elif model == 'Random Forest':
    x1 = pred_rf

x2 = ['You lost him/her 😭' if x1[0] == 'Left' else 'Relax, s/he is still yours 🥳']

if st.button('Predict'):
    if model == 'Gradient Boosting':
        st.metric('Gradient Boosting Prediction', value=x2[0])
    elif model == 'K-Nearest Neighbors':
        st.metric('KNN Prediction', value=x2[0])
    elif model == 'Random Forest':
        st.metric('Random Forest Prediction', value=x2[0])

    if x1[0] == 'Left':
        st.image('dont_go.gif', width=500)
    else:
        st.image('forever.gif', width=500)

# To hide Streamlit style
hide_st_style = """
        <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        header {visibility:hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)

# to add a background image
@st.cache(allow_output_mutation=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = """
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
    """ % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg('background_image.png')