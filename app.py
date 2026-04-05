"""
Student Dropout Risk Prediction — Advisor Dashboard
ITEC 6240 - Machine Learning and Its Applications
Hammad Mirza
"""
import streamlit as st
import pandas as pd
import numpy as np
import json, shap
import plotly.graph_objects as go
import plotly.express as px
import matplotlib; matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')

MARITAL_STATUS = {1:"Single",2:"Married",3:"Widower",4:"Divorced",5:"Common-law union",6:"Legally separated"}
NATIONALITY = {1:"Portuguese",2:"German",6:"English",11:"Mozambican",13:"Turkish",14:"Brazilian",17:"Mexican",21:"Colombian",22:"Ukrainian",24:"Russian",25:"Moldovan",26:"Cuban",32:"Romanian",41:"Lithuanian",62:"Cape Verdean",100:"Angolan",101:"Spanish",103:"Guinean",105:"Santomean",108:"Dutch",109:"Italian"}
APPLICATION_MODE = {1:"1st phase - general contingent",2:"Ordinance No. 612/93",5:"Ordinance No. 854-B/99",7:"1st phase - Madeira Island",10:"Ordinance 533-A/99 b2 (Diff. Plan)",15:"International student (bachelor)",16:"1st phase - Azores Island",17:"Short cycle diploma holders",18:"Change of institution/course",26:"Tech. specialization diploma",27:"Change inst./course (international)",39:"Over 23 years old",42:"Transfer",43:"Change of course",44:"Holders of other higher courses",51:"2nd phase - general contingent",53:"3rd phase - general contingent",57:"Ordinance 533-A/99 b3 (Other Inst.)"}
COURSE = {33:"Biofuel Production Technologies",171:"Animation and Multimedia Design",8014:"Social Service (evening)",9003:"Agronomy",9070:"Communication Design",9085:"Veterinary Nursing",9119:"Informatics Engineering",9130:"Equiniculture",9147:"Management",9238:"Social Service",9254:"Tourism",9500:"Nursing",9556:"Oral Hygiene",9670:"Advertising & Marketing Mgmt",9773:"Journalism and Communication",9853:"Basic Education",9991:"Management (evening)"}
PREV_QUALIFICATION = {1:"Secondary education",2:"Higher ed - bachelor's",3:"Higher ed - degree",4:"Higher ed - master's",5:"Higher ed - doctorate",6:"Frequency of higher ed",9:"12th year (not completed)",10:"11th year (not completed)",12:"Other - 11th year",14:"10th year",15:"10th year (not completed)",19:"Basic ed (3rd cycle/9th year)",38:"Basic ed (2nd cycle/6th year)",39:"Tech. specialization course",40:"Higher ed - degree (1st cycle)",42:"Professional higher tech. course",43:"Higher ed - master's (2nd cycle)"}
PARENT_QUALIFICATION = {1:"Secondary Ed (12th Year)",2:"Higher Ed - Bachelor's",3:"Higher Ed - Degree",4:"Higher Ed - Master's",5:"Higher Ed - Doctorate",6:"Frequency of Higher Ed",9:"12th Year (Not Completed)",10:"11th Year (Not Completed)",11:"7th Year (Old System)",12:"Other - 11th Year",13:"2nd year complementary HS",14:"10th Year",18:"General commerce course",19:"Basic Ed (3rd Cycle/9th Year)",20:"Complementary accounting/admin",22:"Technical-professional course",25:"Complementary theology/arts",26:"7th Year of Schooling",27:"2nd Cycle General High School",29:"9th Year (Not Completed)",30:"8th Year of Schooling",31:"General Admin and Commerce",33:"Supplementary Accounting/Admin",34:"Unknown",35:"Cannot read or write",36:"Can read without 4th year",37:"Basic Ed (1st Cycle/4th Year)",38:"Basic Ed (2nd Cycle/6th Year)",39:"Tech. Specialization Course",40:"Higher Ed - Degree (1st Cycle)",41:"Specialized higher studies",42:"Professional Higher Tech. Course",43:"Higher Ed - Master's (2nd Cycle)",44:"Higher Ed - Doctorate (3rd Cycle)"}
PARENT_OCCUPATION = {0:"Student",1:"Directors/Executives/Managers",2:"Intellectual & Scientific Specialists",3:"Intermediate Level Technicians",4:"Administrative Staff",5:"Service/Security/Sales Workers",6:"Farmers/Skilled Agri Workers",7:"Skilled Industry/Construction Workers",8:"Machine Operators/Assembly Workers",9:"Unskilled Workers",10:"Armed Forces",90:"Other Situation",99:"(blank)",101:"Armed Forces Officers",102:"Armed Forces Sergeants",103:"Other Armed Forces",112:"Admin/commercial directors",114:"Hotel/catering/trade directors",121:"Physical/math/engineering specialists",122:"Health professionals",123:"Teaching professionals",124:"Finance/accounting/admin specialists",125:"ICT Specialists",131:"Science/engineering technicians",132:"Health technicians",134:"Legal/social/cultural technicians",135:"ICT technicians",141:"Office workers/secretaries",143:"Accounting/statistical/financial operators",144:"Other admin support",151:"Personal service workers",152:"Sellers",153:"Personal care workers",154:"Protection/security services",161:"Market-oriented farmers",163:"Subsistence farmers/fishers",171:"Skilled construction workers",172:"Skilled metal/metalworking workers",173:"Crafts/printing workers",174:"Skilled electrical/electronics workers",175:"Food/woodworking/garment workers",181:"Fixed plant/machine operators",182:"Assembly workers",183:"Vehicle drivers/mobile equipment",191:"Cleaning workers",192:"Unskilled agri/forestry/fishery",193:"Unskilled industry/construction",194:"Meal preparation assistants",195:"Street vendors/other services"}
DISPLAY_NAMES = {'Curricular units 2nd sem (approved)':'2nd Sem Courses Approved','Curricular units 2nd sem (grade)':'2nd Sem Average Grade','Curricular units 1st sem (approved)':'1st Sem Courses Approved','Curricular units 1st sem (grade)':'1st Sem Average Grade','Tuition fees up to date':'Tuition Payment Status','Age at enrollment':'Age at Enrollment','Admission grade':'Admission Grade','Scholarship holder':'Scholarship Holder',"Mother's qualification":"Mother's Education","Father's qualification":"Father's Education",'Curricular units 2nd sem (enrolled)':'2nd Sem Enrolled','Curricular units 1st sem (enrolled)':'1st Sem Enrolled','Curricular units 2nd sem (evaluations)':'2nd Sem Evaluations','Curricular units 1st sem (evaluations)':'1st Sem Evaluations','Curricular units 2nd sem (without evaluations)':'2nd Sem Without Eval','Curricular units 1st sem (without evaluations)':'1st Sem Without Eval','Curricular units 2nd sem (credited)':'2nd Sem Credited','Curricular units 1st sem (credited)':'1st Sem Credited','Application mode':'Application Mode','Daytime/evening attendance':'Attendance','Previous qualification (grade)':'Prev. Qual. Grade','Marital status':'Marital Status','Application order':'Application Order','Course':'Course','Displaced':'Displaced','Gender':'Gender','Nacionality':'Nationality','International':'International',"Mother's occupation":"Mother's Occupation","Father's occupation":"Father's Occupation",'Educational special needs':'Special Needs','Debtor':'Debtor','Previous qualification':'Prev. Qualification','Unemployment rate':'Unemployment Rate','Inflation rate':'Inflation Rate','GDP':'GDP'}

st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.main-header{background:linear-gradient(135deg,#1a365d 0%,#2b6cb0 100%);padding:24px 28px;border-radius:12px;margin-bottom:20px;color:white;}
.main-header h1{color:white;margin:0;font-size:1.7em;}.main-header p{color:#bee3f8;margin:6px 0 0 0;font-size:0.98em;}
.info-box{background:#ebf8ff;border-left:4px solid #2b6cb0;padding:12px 15px;border-radius:0 8px 8px 0;margin:10px 0;font-size:0.90em;color:#1a365d;}
.warn-box{background:#fffff0;border-left:4px solid #d69e2e;padding:12px 15px;border-radius:0 8px 8px 0;margin:10px 0;font-size:0.90em;color:#744210;}
.risk-high{background:#fff5f5;border:2px solid #e53e3e;border-radius:12px;padding:22px;text-align:center;}
.risk-medium{background:#fffff0;border:2px solid #d69e2e;border-radius:12px;padding:22px;text-align:center;}
.risk-low{background:#f0fff4;border:2px solid #38a169;border-radius:12px;padding:22px;text-align:center;}
.disclaimer{background:#f7fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;margin:14px 0;font-size:0.83em;color:#4a5568;line-height:1.5;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_or_train():
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    df = pd.read_csv(data_path, sep=';')
    df.columns = [c.strip() for c in df.columns]
    le_ = LabelEncoder()
    df['Target_encoded'] = le_.fit_transform(df['Target'])
    X = df.drop(columns=['Target','Target_encoded']); y = df['Target_encoded']
    fn_ = list(X.columns); cl_ = list(le_.classes_)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    sc_ = StandardScaler()
    Xtr_s = sc_.fit_transform(Xtr); Xte_s = sc_.transform(Xte)
    sm = SMOTE(random_state=42)
    Xtr_sm,ytr_sm = sm.fit_resample(Xtr_s,ytr)
    rf_ = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
    rf_.fit(Xtr_sm,ytr_sm)
    base = [('rf',RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)),('lr',LogisticRegression(max_iter=1000,random_state=42)),('xgb',XGBClassifier(n_estimators=200,random_state=42,eval_metric='mlogloss'))]
    stk_ = StackingClassifier(estimators=base,final_estimator=LogisticRegression(max_iter=1000,random_state=42),cv=5,stack_method='predict_proba',n_jobs=-1)
    stk_.fit(Xtr_sm,ytr_sm)
    exp_ = shap.TreeExplainer(rf_)
    lt_path = os.path.join(os.path.dirname(__file__), 'lms_thresholds.json')
    with open(lt_path) as f: lt_ = json.load(f)
    return rf_, stk_, sc_, le_, fn_, cl_, exp_, lt_

try:
    with st.spinner("Training models on first run (~30 seconds, cached after that)..."):
        rf, stacking, scaler, le, feature_names, classes, explainer, lms_thresh = load_or_train()
    OK = True
except Exception as e:
    OK = False; st.error(f"Cannot train models: {e}")

with st.sidebar:
    st.markdown("## 🎓 Dropout Predictor")
    st.markdown("---")
    mode=st.radio("Navigation",["📊 Individual Prediction","📁 Batch Upload","ℹ️ About"],label_visibility="collapsed")
    st.markdown("---")
    model_choice=st.selectbox("Model",["Stacking Ensemble (Best)","Random Forest"],help="Stacking Ensemble: 78.5% accuracy, 0.732 macro F1 — recommended.\nRandom Forest: 77.2% accuracy, 0.718 macro F1.")
    st.markdown("---")
    st.caption("ITEC 6240 — York University, 2026")
    st.caption("Hammad Mirza")

def get_model(): return stacking if "Stacking" in model_choice else rf
def predict_student(fd):
    df=pd.DataFrame([fd])[feature_names]; Xs=scaler.transform(df.values); m=get_model()
    pi=m.predict(Xs)[0]; pr=m.predict_proba(Xs)[0]
    pd_={classes[i]:round(float(pr[i]),4) for i in range(len(classes))}
    sv=explainer.shap_values(Xs); sf=sv[0,:,pi]
    return classes[pi],pd_,sf
def risk_html(label,conf):
    css={"Dropout":"risk-high","Enrolled":"risk-medium"}.get(label,"risk-low")
    emo={"Dropout":"🔴","Enrolled":"🟡"}.get(label,"🟢")
    col={"Dropout":"#e53e3e","Enrolled":"#d69e2e"}.get(label,"#38a169")
    return f'<div class="{css}"><h1 style="color:{col};margin:0;font-size:2.3em;">{emo} {label}</h1><p style="font-size:1.1em;color:{col};margin:5px 0 0 0;">Confidence: {conf*100:.1f}%</p></div>'
def assess_lms(days,assigns,wifi,minutes):
    results=[]
    for name,val,key in [("LMS Login Days/Month",days,'lms_days_monthly'),("Assignments Submitted/Month",assigns,'lms_assigns_monthly'),("Campus Wi-Fi Days/Month",wifi,'wifi_days_monthly'),("LMS Minutes/Month",minutes,'lms_minutes_monthly')]:
        t=lms_thresh[key]
        if val<=t['high_risk']: results.append({"metric":name,"value":val,"level":"HIGH CONCERN","color":"#e53e3e","note":f"At or below dropout median ({t['high_risk']}). Dropout avg: {t['dropout_mean']}/mo, Non-dropout avg: {t['non_dropout_mean']}/mo"})
        elif val>=t['low_risk']: results.append({"metric":name,"value":val,"level":"LOW CONCERN","color":"#38a169","note":f"At or above non-dropout median ({t['low_risk']})"})
        else: results.append({"metric":name,"value":val,"level":"MODERATE CONCERN","color":"#d69e2e","note":f"Between dropout median ({t['high_risk']}) and non-dropout median ({t['low_risk']})"})
    return results

# ================================================================
# INDIVIDUAL PREDICTION
# ================================================================
if mode=="📊 Individual Prediction" and OK:
    st.markdown('<div class="main-header"><h1>📊 Individual Student Risk Assessment</h1><p>Predict whether a student is likely to drop out, remain enrolled, or graduate — based on academic performance, financial status, and demographics.</p></div>', unsafe_allow_html=True)

    st.markdown("""<div class="warn-box">
    <strong>⚠️ Model context:</strong> Trained on <strong>4,424 students from a Portuguese polytechnic</strong> (2008–2019).
    All grades use a <strong>0–20 Portuguese scale</strong> (passing = 10) and admission scores use a <strong>0–200 scale</strong>.
    Cross-institutional validation with 20,427 Spanish students confirmed that the types of features that predict dropout generalize across institutions —
    but the specific scales require conversion. Use the Canadian conversion guide below when entering data from a Canadian institution.
    </div>""", unsafe_allow_html=True)

    with st.expander("🍁 Canadian Grade Conversion Guide", expanded=False):
        st.markdown("""
**How to convert Canadian grades to the Portuguese scales used by this model**

---

**Course Grades → Portuguese 0–20 scale**

| Canadian % | Letter Grade | → Portuguese /20 |
|---|---|---|
| 90–100% | A+ | 18–20 |
| 80–89% | A / A− | 16–18 |
| 70–79% | B+ / B | 14–16 |
| 60–69% | C+ / C | 12–14 |
| 50–59% | D | 10–12 |
| < 50% | F | < 10 |

**Formula: Divide your Canadian percentage by 5**
> 75% → 75 ÷ 5 = **15.0** &nbsp;|&nbsp; 60% → 60 ÷ 5 = **12.0** &nbsp;|&nbsp; 50% → 50 ÷ 5 = **10.0** (minimum pass)

*Portuguese passing = 10/20. Canadian passing = 50%. The scales are directly proportional.*

> ⚠️ **Important grading culture note:** This linear conversion is a reasonable approximation for ML feature input,
> but Portuguese professors rarely award grades above 17–18/20. A 15/20 in Portugal is considered "Good" and reflects
> stronger relative performance than the raw number suggests — in practice it sits closer to the top 20–25% of students.
> By contrast, an A in Canada (80–89%) converts to 16–18 on this scale, which would be exceptional by Portuguese standards.
> **For entering data from Canadian students, the linear formula is correct and standard — just be aware that
> a converted grade of 16+ does not mean the Portuguese model "sees" that as easy to achieve.**

---

**Admission Grade → Portuguese 0–200 scale**

The Portuguese admission grade is the Canadian percentage on a doubled scale.

| Canadian HS / Admission Average | → Portuguese /200 |
|---|---|
| 95% (Ontario top admit) | 190 |
| 90% | 180 |
| 85% | 170 |
| 80% | 160 |
| 75% | 150 |
| 70% | 140 |

**Formula: Multiply your Canadian percentage by 2**
> Ontario 85% average → 85 × 2 = **170** &nbsp;|&nbsp; 78% → 78 × 2 = **156**

Dataset average admission grade: **127 / 200** (~63.5% in Canadian terms — typical for a polytechnic program).

---

*GPA note: Canadian GPA scales vary by institution (4.0 at York/U of T, 4.33 at UBC, 9-point at some Ontario schools, 4.3 at U of A).
The percentage-based formula above works across all Canadian institutions and maps directly to this model's Portuguese scale.*
        """)

    st.markdown("---")
    st.markdown("### ⭐ Key Risk Indicators — ~79% of Prediction")
    st.caption("Three groups that together account for ~79% of what the model uses. Fill what you know — each group adds independent predictive value.")

    st.markdown('<div style="background:#1a365d;color:white;padding:7px 13px;border-radius:6px;font-weight:700;font-size:0.93em;margin-top:12px;margin-bottom:6px;">📚 Group A — Semester Performance &nbsp;<span style="font-weight:400;font-size:0.84em;">(SHAP #1 #2 #3 #8 · ~48% of prediction)</span></div>', unsafe_allow_html=True)
    st.caption("Courses passed and average grades for both semesters. These 4 fields alone drive nearly half the prediction.")
    ga1, ga2 = st.columns(2)
    with ga1:
        s2a = st.number_input("2nd Sem — Courses Passed", 0, 30, 5, key="s2a", help="SHAP #1 — strongest predictor. Number of courses the student successfully passed in semester 2. Dropouts avg 1.94; graduates avg 6.18. Typical full-time load: 6 courses.")
        s2g = st.number_input("2nd Sem — Average Grade (0–20)", 0.0, 20.0, 12.0, 0.1, key="s2g", help="SHAP #3. Average grade across all semester 2 courses. Portuguese scale: 0–20, passing = 10. Dropouts avg 5.9; graduates avg 12.7. 🇨🇦 Divide Canadian % by 5.")
    with ga2:
        s1a = st.number_input("1st Sem — Courses Passed", 0, 30, 5, key="s1a", help="SHAP #2. Number of courses passed in semester 1. A strong semester 1 is a positive retention signal, though semester 2 is more predictive.")
        s1g = st.number_input("1st Sem — Average Grade (0–20)", 0.0, 20.0, 12.0, 0.1, key="s1g", help="SHAP #8. Average grade across semester 1 courses. Same 0–20 scale. 🇨🇦 Divide Canadian % by 5.")

    st.markdown('<div style="background:#2c7a7b;color:white;padding:7px 13px;border-radius:6px;font-weight:700;font-size:0.93em;margin-top:14px;margin-bottom:6px;">🎯 Group B — Course Engagement &nbsp;<span style="font-weight:400;font-size:0.84em;">(SHAP #5 #6 #10 #11 · +16% of prediction)</span></div>', unsafe_allow_html=True)
    st.caption("Courses enrolled in vs. exams actually attempted. Low participation signals disengagement independently of grades.")
    gb1, gb2 = st.columns(2)
    with gb1:
        s1v = st.number_input("1st Sem — Exams Attempted", 0, 50, 6, key="s1v", help="SHAP #5. Courses where the student sat for at least one exam in semester 1. A student can be enrolled and never show up — the gap between enrolled and attempted is a disengagement signal.")
        s1e = st.number_input("1st Sem — Courses Enrolled", 0, 30, 6, key="s1e", help="SHAP #11. Total courses registered in semester 1. Typical full-time load: 5–6 courses.")
    with gb2:
        s2v = st.number_input("2nd Sem — Exams Attempted", 0, 50, 6, key="s2v", help="SHAP #6. Courses where the student sat for at least one exam in semester 2. A drop from semester 1 is a strong dropout signal.")
        s2e = st.number_input("2nd Sem — Courses Enrolled", 0, 30, 6, key="s2e", help="SHAP #10. Total courses registered in semester 2.")
    st.caption("💡 If exams attempted < courses enrolled, the student is registering but not participating — an early warning of disengagement.")

    st.markdown('<div style="background:#744210;color:white;padding:7px 13px;border-radius:6px;font-weight:700;font-size:0.93em;margin-top:14px;margin-bottom:6px;">💰 Group C — Financial & Personal &nbsp;<span style="font-weight:400;font-size:0.84em;">(SHAP #4 #7 #9 · +15% of prediction)</span></div>', unsafe_allow_html=True)
    st.caption("Tuition status, scholarship, and age. Tuition becomes the #1 predictor (SHAP 0.184) for students already in financial distress.")
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        tuition = st.selectbox("Tuition Fees Up to Date", [(1,"Yes — current"),(0,"No — overdue")], format_func=lambda x:x[1], key="tuition", help="SHAP #4. Whether tuition is fully paid. Only 67.8% of dropouts were current vs 98.7% of graduates. For students behind on tuition, this feature jumps to SHAP #1 (0.184) — the single strongest predictor in the entire model.")
    with gc2:
        scholarship = st.selectbox("Scholarship Holder", [(0,"No"),(1,"Yes")], format_func=lambda x:x[1], key="scholarship_key", help="SHAP #7. Whether the student holds a scholarship. Acts as a protective factor. ⚠️ Fairness warning: model misses 53% of scholarship-holder dropouts — always follow up directly with this group regardless of model output.")
    with gc3:
        age = st.number_input("Age at Enrollment", 17, 70, 20, key="age_key", help="SHAP #9. Age when the student first enrolled. Students 30+ show significantly higher dropout risk, likely due to competing work and family obligations.")
    st.markdown("""<div class="warn-box" style="font-size:0.85em;padding:8px 12px;margin-top:6px;">
    <strong>⚠️ Scholarship-holder fairness warning:</strong> The model detects only <strong>47%</strong> of scholarship-holder dropouts vs 74% for non-holders.
    Always follow up directly with scholarship students showing declining performance, regardless of model output.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Additional Details")
    st.caption("These fields contribute the remaining ~21% of predictive power. The model works without them, but accuracy improves with more complete data.")

    with st.expander("📚 Additional Semester Details", expanded=False):
        st.caption("Credited (transferred) units and units with no exam attempt. SHAP ranks #20+. Only fill if available.")
        ec1,ec2=st.columns(2)
        with ec1:
            st.markdown("**1st Semester**")
            s1c=st.number_input("Credited (transferred units)",0,30,0,key="s1c",help="Units credited from prior learning or transferred from another program. These do not count toward the student's regular academic load.")
            s1w=st.number_input("Without Evaluations",0,20,0,key="s1w",help="Units enrolled in but no exam was ever attempted. High values indicate the student stopped participating before assessments.")
        with ec2:
            st.markdown("**2nd Semester**")
            s2c=st.number_input("Credited (transferred units)",0,30,0,key="s2c",help="Transferred or credited units in semester 2.")
            s2w=st.number_input("Without Evaluations",0,20,0,key="s2w",help="Units enrolled but no exam attempted in semester 2. A rise from semester 1 is a clear disengagement signal.")

    with st.expander("👤 Demographics & Personal Information", expanded=False):
        st.caption("Age and scholarship are in Key Risk Indicators above. Remaining fields have low-to-moderate predictive impact.")
        d1,d2,d3=st.columns(3)
        with d1:
            gender=st.selectbox("Gender",[(1,"Male"),(0,"Female")],format_func=lambda x:x[1],help="The model detects dropouts at nearly equal rates across genders (72.4% male vs 72.7% female recall), though males are falsely flagged at a slightly higher rate (10% vs 5.8%).")
            marital=st.selectbox("Marital Status",[(k,v) for k,v in sorted(MARITAL_STATUS.items())],format_func=lambda x:x[1],help="88% of dataset students are Single. Moderate predictive impact for other categories.")
        with d2:
            nationality=st.selectbox("Nationality",[(k,v) for k,v in sorted(NATIONALITY.items(),key=lambda x:x[1])],format_func=lambda x:x[1],help="93% of dataset is Portuguese. Other nationalities have very low predictive impact due to small sample size.")
            displaced=st.selectbox("Displaced from Home",[(0,"No - lives locally"),(1,"Yes - relocated to attend")],format_func=lambda x:x[1],help="Whether the student moved from their home region to attend. Displaced students actually graduate at slightly higher rates in this dataset.")
            international=st.selectbox("International Student",[(0,"No"),(1,"Yes")],format_func=lambda x:x[1],help="International admission pathway. Very low SHAP importance.")
        with d3:
            special_needs=st.selectbox("Educational Special Needs",[(0,"No"),(1,"Yes")],format_func=lambda x:x[1],help="Whether the student has registered educational special needs. Very low predictive impact in this dataset.")

    with st.expander("💰 Socioeconomic & Financial", expanded=False):
        st.caption("Scholarship moved to Key Risk Indicators. Debtor status and parent background have low-to-moderate impact.")
        s1_,s2_,s3_=st.columns(3)
        with s1_:
            debtor=st.selectbox("Debtor",[(0,"No"),(1,"Yes")],format_func=lambda x:x[1],help="Whether the student has outstanding debts to the institution beyond tuition. Adds moderate dropout signal.")
        with s2_:
            mother_qual=st.selectbox("Mother's Qualification",[(k,v) for k,v in sorted(PARENT_QUALIFICATION.items(),key=lambda x:x[1])],format_func=lambda x:x[1],help="Mother's highest level of education (Portuguese classification). Select the closest equivalent.")
            father_qual=st.selectbox("Father's Qualification",[(k,v) for k,v in sorted(PARENT_QUALIFICATION.items(),key=lambda x:x[1])],format_func=lambda x:x[1],help="Father's highest education level. First-generation students (parents without university education) show higher dropout rates.")
        with s3_:
            mother_occ=st.selectbox("Mother's Occupation",[(k,v) for k,v in sorted(PARENT_OCCUPATION.items(),key=lambda x:x[1])],format_func=lambda x:x[1],help="Mother's occupation using the Portuguese Classification of Occupations. Select the closest match.")
            father_occ=st.selectbox("Father's Occupation",[(k,v) for k,v in sorted(PARENT_OCCUPATION.items(),key=lambda x:x[1])],format_func=lambda x:x[1],help="Father's occupation. Low individual impact, but combined with parent education provides socioeconomic context.")

    with st.expander("🏫 Enrollment Context", expanded=False):
        st.caption("Admission grade is SHAP #17. Course program is SHAP #12 — Nursing and Social Service have the highest graduation rates.")
        e1_,e2_,e3_=st.columns(3)
        with e1_:
            admission_grade=st.number_input("Admission Grade (0–200)",0.0,200.0,130.0,0.5,key="adm_grade",help="Entry grade on the Portuguese 0–200 scale. Dataset average: 127. 🇨🇦 Multiply Canadian % by 2 (e.g., Ontario 85% → 170).")
            app_mode=st.selectbox("Application Mode",[(k,v) for k,v in sorted(APPLICATION_MODE.items(),key=lambda x:x[1])],format_func=lambda x:x[1],help="Admission pathway. '1st phase - general contingent' is standard competitive entry. Transfer students have slightly higher dropout rates.")
            app_order=st.number_input("Application Order (0=1st choice, 9=last)",0,9,1,help="Student's program preference rank (0=first choice). Students admitted to their first-choice program have better retention.")
        with e2_:
            course=st.selectbox("Course / Program",[(k,v) for k,v in sorted(COURSE.items(),key=lambda x:x[1])],format_func=lambda x:x[1],help="Degree program. Nursing and Social Service have the highest graduation rates. Biofuel Production and Informatics Engineering have the highest dropout rates.")
        with e3_:
            prev_qual=st.selectbox("Previous Qualification",[(k,v) for k,v in sorted(PREV_QUALIFICATION.items(),key=lambda x:x[1])],format_func=lambda x:x[1],help="Student's highest qualification before enrolling. Secondary education is the most common. Prior higher education indicates a returning/transfer student.")
            prev_qual_grade=st.number_input("Previous Qualification Grade (0–200)",0.0,200.0,130.0,0.5,help="Grade from the previous qualification. Same 0–200 scale as admission grade. 🇨🇦 Multiply Canadian % by 2.")
            daytime=st.selectbox("Attendance",[(1,"Daytime"),(0,"Evening")],format_func=lambda x:x[1],help="Daytime students graduate at higher rates than evening students, likely because evening students have more competing obligations.")

    with st.expander("🌍 Macroeconomic (near-zero predictive impact)", expanded=False):
        st.caption("GDP, unemployment, and inflation ranked last in SHAP. All students in the same cohort year share identical values, so these features carry no discriminative power between individuals. Safe to leave at defaults.")
        m1_,m2_,m3_=st.columns(3)
        unemp=m1_.number_input("Unemployment Rate (%)",value=10.8,step=0.1,help="National unemployment rate at time of enrollment. Dataset range: 7.6%–16.2%.")
        infl=m2_.number_input("Inflation Rate (%)",value=1.4,step=0.1,help="National inflation rate at time of enrollment.")
        gdp=m3_.number_input("GDP Growth Rate",value=1.74,step=0.01,help="GDP growth rate at time of enrollment. All three macroeconomic indicators have near-zero SHAP values.")

    with st.expander("📱 LMS Engagement Assessment (optional — supplementary behavioral check)", expanded=False):
        st.markdown("""<div class="info-box">
        <strong>What is this?</strong> If your institution tracks LMS activity (eClass, Moodle, Canvas, Blackboard),
        enter <strong>monthly averages</strong> here. Thresholds come from <strong>20,427 students at a Spanish university (UPV, 2022)</strong> and
        represent the most universally applicable component of this tool — LMS metrics are platform-agnostic and don't depend on any grading system.<br><br>
        This runs <strong>independently</strong> of the main prediction model as a supplementary behavioral check.
        </div>""", unsafe_allow_html=True)
        l1_,l2_=st.columns(2)
        lms_days=l1_.number_input("LMS Login Days/Month",0.0,31.0,0.0,0.5,help=f"Average days/month the student logged into the course platform. Dropout avg: {lms_thresh['lms_days_monthly']['dropout_mean']}/mo, Non-dropout avg: {lms_thresh['lms_days_monthly']['non_dropout_mean']}/mo")
        lms_assigns=l1_.number_input("Assignment Submissions/Month",0.0,50.0,0.0,0.5,help=f"Average assignment submissions per month. Dropout avg: {lms_thresh['lms_assigns_monthly']['dropout_mean']}/mo, Non-dropout avg: {lms_thresh['lms_assigns_monthly']['non_dropout_mean']}/mo")
        wifi_d=l2_.number_input("Campus Wi-Fi Days/Month",0.0,31.0,0.0,0.5,help=f"Average days/month student device detected on campus Wi-Fi. Dropout avg: {lms_thresh['wifi_days_monthly']['dropout_mean']}/mo, Non-dropout avg: {lms_thresh['wifi_days_monthly']['non_dropout_mean']}/mo")
        lms_mins=l2_.number_input("LMS Total Minutes/Month",0.0,10000.0,0.0,10.0,help=f"Average total minutes/month on the LMS platform. Dropout avg: {lms_thresh['lms_minutes_monthly']['dropout_mean']} min/mo, Non-dropout avg: {lms_thresh['lms_minutes_monthly']['non_dropout_mean']} min/mo")
        lms_entered=any([lms_days>0,lms_assigns>0,wifi_d>0,lms_mins>0])

    st.markdown("---")
    features={'Marital status':marital[0],'Application mode':app_mode[0],'Application order':app_order,'Course':course[0],'Daytime/evening attendance':daytime[0],'Previous qualification':prev_qual[0],'Previous qualification (grade)':prev_qual_grade,'Nacionality':nationality[0],"Mother's qualification":mother_qual[0],"Father's qualification":father_qual[0],"Mother's occupation":mother_occ[0],"Father's occupation":father_occ[0],'Admission grade':admission_grade,'Displaced':displaced[0],'Educational special needs':special_needs[0],'Debtor':debtor[0],'Tuition fees up to date':tuition[0],'Gender':gender[0],'Scholarship holder':scholarship[0],'Age at enrollment':age,'International':international[0],'Curricular units 1st sem (credited)':s1c,'Curricular units 1st sem (enrolled)':s1e,'Curricular units 1st sem (evaluations)':s1v,'Curricular units 1st sem (approved)':s1a,'Curricular units 1st sem (grade)':s1g,'Curricular units 1st sem (without evaluations)':s1w,'Curricular units 2nd sem (credited)':s2c,'Curricular units 2nd sem (enrolled)':s2e,'Curricular units 2nd sem (evaluations)':s2v,'Curricular units 2nd sem (approved)':s2a,'Curricular units 2nd sem (grade)':s2g,'Curricular units 2nd sem (without evaluations)':s2w,'Unemployment rate':unemp,'Inflation rate':infl,'GDP':gdp}

    if st.button("🔍 **Predict Dropout Risk**",type="primary",use_container_width=True):
        pred_label,proba_dict,shap_for_pred=predict_student(features)
        st.markdown("---")
        _,col_r,_=st.columns([1,2,1])
        with col_r: st.markdown(risk_html(pred_label,proba_dict[pred_label]),unsafe_allow_html=True)
        st.markdown("#### Outcome Probabilities")
        cmap={'Dropout':'#e53e3e','Enrolled':'#d69e2e','Graduate':'#38a169'}
        fig=go.Figure()
        for o in classes: fig.add_trace(go.Bar(x=[proba_dict[o]],y=[o],orientation='h',marker_color=cmap[o],text=f"{proba_dict[o]*100:.1f}%",textposition='auto'))
        fig.update_layout(showlegend=False,height=175,xaxis=dict(range=[0,1],tickformat='.0%'),margin=dict(l=0,r=0,t=10,b=10),yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig,use_container_width=True)
        st.markdown(f"#### 🔍 Why This Prediction?")
        st.caption(f"Top factors influencing the prediction toward **{pred_label}**. Red bars increase risk; green bars reduce risk.")
        abs_s=np.abs(shap_for_pred); ti=np.argsort(abs_s)[-10:][::-1]
        sn=[DISPLAY_NAMES.get(feature_names[i],feature_names[i]) for i in ti]; sv_=[shap_for_pred[i] for i in ti]
        fig_s=go.Figure()
        fig_s.add_trace(go.Bar(y=sn[::-1],x=sv_[::-1],orientation='h',marker_color=['#e53e3e' if v>0 else '#38a169' for v in sv_[::-1]],text=[f"{v:+.3f}" for v in sv_[::-1]],textposition='outside'))
        fig_s.update_layout(title=f"Top 10 Feature Contributions → {pred_label}",xaxis_title="SHAP Value",height=400,margin=dict(l=0,r=0,t=40,b=10))
        st.plotly_chart(fig_s,use_container_width=True)
        st.markdown("#### 💡 Recommended Actions")
        if pred_label=="Dropout":
            if s2a<4: st.error("⚠️ **Low 2nd semester completion** — Refer to academic advising to review course load or arrange tutoring.")
            if tuition[0]==0: st.error("⚠️ **Tuition overdue** — Connect with financial aid for payment plans, emergency bursaries, or work-study options.")
            if scholarship[0]==0 and s1g>10: st.warning("💡 **No scholarship but passing grades** — Explore scholarship eligibility; financial support may improve retention.")
            if age>25: st.warning("💡 **Mature student** — Consider flexible scheduling, part-time enrollment, or evening class alternatives.")
            if s2a>=4 and tuition[0]==1: st.info("📋 Academic and financial indicators are moderate — schedule an advising meeting to surface any underlying concerns before they escalate.")
        elif pred_label=="Enrolled":
            st.warning("🟡 Student is on an **uncertain trajectory** — shares characteristics with both graduates and students at risk.")
            st.info("📋 Schedule a check-in within 2 weeks to assess engagement, course satisfaction, and any emerging concerns.")
        else:
            st.success("🟢 Student appears **on track to graduate**. No immediate intervention required.")
            st.info("💡 Consider for peer mentoring, teaching assistant roles, or student leadership opportunities.")
        if lms_entered:
            st.markdown("---"); st.markdown("#### 📱 LMS Behavioral Risk Assessment")
            st.caption("Based on 20,427 students at Universitat Politecnica de Valencia, Spain (2022). Independent of the main prediction model.")
            lr=assess_lms(lms_days,lms_assigns,wifi_d,lms_mins)
            hc=sum(1 for r in lr if r['level']=="HIGH CONCERN")
            if hc>=3: st.error("🔴 **Overall LMS Engagement: HIGH CONCERN** — Multiple indicators below dropout thresholds. Immediate outreach recommended.")
            elif hc>=1: st.warning("🟡 **Overall LMS Engagement: MODERATE CONCERN** — Some indicators below typical thresholds.")
            else: st.success("🟢 **Overall LMS Engagement: LOW CONCERN** — Engagement indicators are within a healthy range.")
            for r in lr: st.markdown(f'<div style="padding:9px 13px;margin:5px 0;border-radius:8px;border-left:5px solid {r["color"]};background:{r["color"]}10;"><strong>{r["metric"]}</strong>: {r["value"]:.1f} → <span style="color:{r["color"]};font-weight:700;">{r["level"]}</span><br><small style="color:#666;">{r["note"]}</small></div>',unsafe_allow_html=True)
        st.markdown("""<div class="disclaimer">
        <strong>Disclaimer:</strong> Academic research prototype — ITEC 6240, York University. Predictions are probabilistic estimates
        based on historical data from a Portuguese institution and are <strong>not</strong> a substitute for professional advising.
        Never use model output as the sole basis for student decisions. For institutional deployment, retrain on local data and review
        with qualified personnel. Data entered here is processed locally and not stored.
        </div>""", unsafe_allow_html=True)

# ================================================================
# BATCH UPLOAD
# ================================================================
elif mode=="📁 Batch Upload" and OK:
    st.markdown('<div class="main-header"><h1>📁 Batch Risk Assessment</h1><p>Upload a CSV of student records to generate dropout risk predictions for an entire cohort.</p></div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>📋 How to use:</strong>
<ol>
<li><strong>Prepare your CSV:</strong> Each row = one student. Must include the same 36 UCI feature columns with exact column names and numeric-coded values. A "Target" column will be ignored if present.</li>
<li><strong>Upload:</strong> Use the file uploader below (semicolon or comma separated).</li>
<li><strong>Predict:</strong> Click "Run Predictions" to process all students with the selected model.</li>
<li><strong>Review:</strong> View cohort summary with risk distribution, ranked list of high-risk students, and a pie chart.</li>
<li><strong>Download:</strong> Export results CSV with Predicted Outcome, Dropout Probability, Enrolled Probability, and Graduate Probability added.</li>
</ol>
<strong>Tip:</strong> Test with the original <code>data.csv</code> file to see batch mode in action.</div>""", unsafe_allow_html=True)
    st.markdown("""<div class="warn-box"><strong>⚠️ Format note:</strong> CSV must use original UCI column names with numeric-coded values (not labels).
    If preparing data from a Canadian institution, convert grades and scores using the guide in the Individual Prediction tab first.
    </div>""", unsafe_allow_html=True)
    uploaded=st.file_uploader("**Upload Student Records CSV**",type=['csv'],help="CSV with 36 UCI features. Max ~50,000 rows recommended.")
    if uploaded:
        try:
            try:
                bdf=pd.read_csv(uploaded,sep=';')
                if len(bdf.columns)<5: uploaded.seek(0); bdf=pd.read_csv(uploaded,sep=',')
            except: uploaded.seek(0); bdf=pd.read_csv(uploaded,sep=',')
            bdf.columns=[c.strip() for c in bdf.columns]
            has_t='Target' in bdf.columns
            bdf_c=bdf.drop(columns=['Target']) if has_t else bdf.copy()
            st.success(f"Loaded **{len(bdf):,}** student records with **{len(bdf_c.columns)}** features.")
            with st.expander("Preview data (first 5 rows)",expanded=False): st.dataframe(bdf.head(),use_container_width=True)
            if st.button("🔍 **Run Predictions**",type="primary",use_container_width=True):
                with st.spinner("Processing..."):
                    mdl=get_model(); Xb=scaler.transform(bdf_c[feature_names].values)
                    pds_=mdl.predict(Xb); pbs_=mdl.predict_proba(Xb)
                    res=bdf.copy()
                    res['Predicted Outcome']=[classes[p] for p in pds_]
                    res['Dropout Probability']=[round(p[0],4) for p in pbs_]
                    res['Enrolled Probability']=[round(p[1],4) for p in pbs_]
                    res['Graduate Probability']=[round(p[2],4) for p in pbs_]
                st.markdown("---"); st.markdown("### Cohort Risk Summary")
                pc=res['Predicted Outcome'].value_counts()
                dn=pc.get('Dropout',0); en=pc.get('Enrolled',0); gn=pc.get('Graduate',0)
                c1,c2,c3=st.columns(3)
                c1.metric("🔴 At Risk (Dropout)",f"{dn:,}",f"{dn/len(res)*100:.1f}%")
                c2.metric("🟡 Uncertain (Enrolled)",f"{en:,}",f"{en/len(res)*100:.1f}%")
                c3.metric("🟢 On Track (Graduate)",f"{gn:,}",f"{gn/len(res)*100:.1f}%")
                fig=px.pie(values=[dn,en,gn],names=['Dropout','Enrolled','Graduate'],color_discrete_map={'Dropout':'#e53e3e','Enrolled':'#d69e2e','Graduate':'#38a169'},title='Predicted Outcome Distribution')
                fig.update_layout(height=340); st.plotly_chart(fig,use_container_width=True)
                st.markdown("### 🔴 High-Risk Students (Predicted Dropout)")
                hr=res[res['Predicted Outcome']=='Dropout'].sort_values('Dropout Probability',ascending=False)
                if len(hr)>0:
                    dcols=[c for c in ['Age at enrollment','Gender','Curricular units 2nd sem (approved)','Curricular units 2nd sem (grade)','Tuition fees up to date','Scholarship holder','Predicted Outcome','Dropout Probability'] if c in hr.columns]
                    st.dataframe(hr[dcols].head(25),use_container_width=True)
                else: st.info("No students predicted as Dropout in this cohort.")
                st.download_button("📥 **Download Full Results (CSV)**",res.to_csv(index=False),"dropout_predictions.csv","text/csv",use_container_width=True)
        except Exception as e: st.error(f"Error processing file: {e}")
    st.markdown("""<div class="disclaimer">
    <strong>Disclaimer:</strong> Batch predictions carry the same limitations as individual predictions. This model was trained on Portuguese institutional data.
    Results must be reviewed by qualified academic staff before any student interventions are initiated.
    </div>""", unsafe_allow_html=True)

# ================================================================
# ABOUT
# ================================================================
elif mode=="ℹ️ About":
    st.markdown('<div class="main-header"><h1>ℹ️ About This Application</h1><p>Model performance, feature importance, fairness audit, LMS thresholds, and references.</p></div>', unsafe_allow_html=True)
    st.markdown(f"""
### Purpose

This application predicts whether a student is likely to **drop out**, **remain enrolled** (uncertain trajectory), or **graduate**,
using 36 features covering demographics, socioeconomic background, financial status, and academic performance across two semesters.
Built as a research prototype for academic advisors exploring early warning systems for student retention.

---

### Model Performance (UCI Test Set, n=885)

| Model | Accuracy | Macro F1 | Dropout F1 | Enrolled F1 | Graduate F1 |
|---|---|---|---|---|---|
| **Stacking Ensemble** (default) | **78.5%** | **0.732** | 0.79 | 0.54 | 0.86 |
| Random Forest | 77.2% | 0.718 | 0.77 | 0.52 | 0.86 |
| XGBoost | 76.7% | 0.706 | 0.76 | 0.49 | 0.86 |
| Logistic Regression | 73.6% | 0.697 | 0.75 | 0.50 | 0.83 |

*Macro F1 is the primary metric — a model predicting Graduate for all students would score ~82% accuracy but fail to detect dropouts.*

---

### Most Important Predictors (SHAP, verified from pipeline output)

| Rank | Feature | Mean |SHAP| | Note |
|---|---|---|---|
| #1 | 2nd semester courses approved | 0.105 | Dropouts avg 1.94 passed; graduates avg 6.18 |
| #2 | 1st semester courses approved | 0.050 | Early retention signal |
| #3 | 2nd semester average grade | 0.046 | Dropouts avg 5.9/20; graduates avg 12.7/20 |
| #4 | Tuition fees up to date | 0.030 | Jumps to 0.184 (#1) for students behind on tuition |
| #5 | 1st semester evaluations | 0.025 | Exams attempted — disengagement signal |
| #6 | 2nd semester evaluations | 0.023 | |
| #7 | Scholarship holder | 0.021 | Protective factor — but model under-detects this group |
| #8 | 1st semester average grade | 0.020 | |
| #9 | Age at enrollment | 0.017 | Students 30+ show amplified dropout risk |

---

### Fairness Audit (Random Forest, four-fifths rule: DI < 0.8 = FAIL)

| Protected Attribute | Disparate Impact | Key Finding |
|---|---|---|
| Scholarship holders | 0.183 — FAIL | Model catches only **47%** of holder dropouts vs 74% of non-holders |
| Gender (M vs F) | 0.571 — FAIL | Detection near-equal (72.4% vs 72.7%), but males falsely flagged more |
| Age (< 22 vs 30+) | 0.347 — FAIL | Older students flagged more aggressively in both directions |
| Tuition status | 0.222 — FAIL | 96% detection for overdue students, but 44% false positive rate |

**Recommendation:** Never rely solely on model output. Always supplement with direct student contact, especially for scholarship holders.

---

### Applicability

**Training data:** 4,424 students from Polytechnic Institute of Portalegre, Portugal (2008–2019).

**Cross-institutional validation:** Same methodology tested on 20,427 students at UPV, Spain. Academic performance features generalized well. For Canadian institutions, use the conversion guide in the Individual Prediction tab.

**What transfers directly:** Pipeline architecture, SHAP explainability, fairness methodology, LMS thresholds.

**What requires local adaptation:** Grade/admission scales, course codes, demographic encodings, model weights.

---

### LMS Behavioral Thresholds (UPV, 20,427 students, 2022 — verified from dataset)

| Metric | Dropout Avg/Month | Non-Dropout Avg/Month | HIGH CONCERN Flag |
|---|---|---|---|
| LMS login days | {lms_thresh['lms_days_monthly']['dropout_mean']} days | {lms_thresh['lms_days_monthly']['non_dropout_mean']} days | ≤ {lms_thresh['lms_days_monthly']['high_risk']} days |
| Assignment submissions | {lms_thresh['lms_assigns_monthly']['dropout_mean']} | {lms_thresh['lms_assigns_monthly']['non_dropout_mean']} | ≤ {lms_thresh['lms_assigns_monthly']['high_risk']} |
| Campus Wi-Fi days | {lms_thresh['wifi_days_monthly']['dropout_mean']} days | {lms_thresh['wifi_days_monthly']['non_dropout_mean']} days | ≤ {lms_thresh['wifi_days_monthly']['high_risk']} days |
| LMS total minutes | {lms_thresh['lms_minutes_monthly']['dropout_mean']} min | {lms_thresh['lms_minutes_monthly']['non_dropout_mean']} min | ≤ {lms_thresh['lms_minutes_monthly']['high_risk']} min |

*Students at or below the HIGH CONCERN threshold had a 2.1× higher dropout rate.*

---

### Training Data & References

**Primary:** UCI "Predict Students' Dropout and Academic Success" (Realinho et al., 2022) — 4,424 students, 36 features. CC BY 4.0.

**Cross-validation:** UPV Longitudinal Dataset (Igualde-Saez et al., 2025) — 20,427 students, 28 features + LMS data.

- Realinho, V. et al. (2022). *Data*, 7(11), 146.
- Igualde-Saez, A. et al. (2025). *Data*, 10(10), 162.
- Liu, Z. et al. (2025). Student Dropout Prediction Using Ensemble Learning with SHAP. *JSSPA*, 2(3).
- Kim, S. et al. (2023). *Applied Sciences*, 13(10), 6275.
- Villar, A. & de Andrade, C. (2024). *Discover AI*, 4(2).

---

### Team

Hammad Mirza
ITEC 6240 — Machine Learning and Its Applications, York University, 2026

---
""")
    st.markdown("""<div class="disclaimer">
    <strong>Legal Disclaimer:</strong> Academic research prototype developed for ITEC 6240 at York University.
    Provided for educational and research purposes only. Predictions are probabilistic estimates based on historical data
    from a single Portuguese institution and do not constitute professional academic advising. The developers and York University
    accept no liability for any decisions made using this tool. For institutional deployment, the model must be retrained on
    local data, validated by qualified data scientists, and reviewed by institutional ethics boards.
    Data entered is processed locally and not stored or transmitted.
    </div>""", unsafe_allow_html=True)
