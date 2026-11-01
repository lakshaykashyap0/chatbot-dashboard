# app.py
import joblib
import numpy as np
import streamlit as st

# ==========================================
# 1. Page Configuration (Deploy options enabled)
# ==========================================
st.set_page_config(
    page_title="CampusCore | University Student AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Glassmorphic UI with Deploy Bar Intact)
st.markdown(
    """
<style>
    /* App Base Background & Typography */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0e1626 0%, #050811 95%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #f1f5f9;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.85));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Chat Message Bubbles */
    [data-testid="stChatMessage"] {
        background: rgba(30, 41, 59, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #090e17 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Recommendation Button Styling */
    div.stButton > button {
        background: rgba(30, 41, 59, 0.75);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 12px;
        padding: 10px 14px;
        font-size: 13px;
        text-align: left;
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #2563eb;
        color: #ffffff;
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3);
    }

    /* Action Resource Link Badges */
    .resource-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(37, 99, 235, 0.18);
        color: #60a5fa !important;
        border: 1px solid rgba(59, 130, 246, 0.35);
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 13px;
        text-decoration: none;
        margin-top: 10px;
        font-weight: 500;
        transition: background 0.2s;
    }
    .resource-badge:hover {
        background: rgba(37, 99, 235, 0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. Load ML Artifacts
# ==========================================
@st.cache_resource
def load_model_assets():
  model = joblib.load('student_intent_classifier.pkl')
  vectorizer = joblib.load('tfidf_vectorizer.pkl')
  encoder = joblib.load('label_encoder.pkl')
  return model, vectorizer, encoder


try:
  classifier, vectorizer, label_encoder = load_model_assets()
except Exception as e:
  st.error(
      '⚠️ Model artifacts missing. Ensure student_intent_classifier.pkl,'
      ' tfidf_vectorizer.pkl, and label_encoder.pkl are present in this'
      ' directory.'
  )
  st.stop()


# ==========================================
# 3. Merged Knowledge Base (All 30 Intents)
# ==========================================
KNOWLEDGE_BASE = {
    'fee_inquiry': {
        'title': 'Fees & Accounts Desk',
        'response': (
            'B.Tech annual tuition fee is Rs. 1,35,000 (payable in two semester'
            ' installments). Hostel and mess fees are Rs. 1,12,000 annually.'
            ' Online payment options include UPI, Net Banking, and Challan.'
        ),
        'action_url': 'https://erp.university.edu/accounts/fees',
        'action_label': 'Open Fee Payment Portal',
    },
    'fee_payment': {
        'title': 'Fee Payment Portal',
        'response': (
            'Generate the fee challan from Portal > Fees > Generate Challan and'
            ' pay via bank counter, debit card, or online net banking.'
        ),
        'action_url': 'https://erp.university.edu/fee-payment',
        'action_label': 'Generate Challan / Pay',
    },
    'schedule': {
        'title': 'Timetable & Academic Slots',
        'response': (
            'Core theory lectures run Monday through Friday from 09:00 AM to'
            ' 05:00 PM. Lab practical slots and elective batches are listed'
            ' section-wise under your student schedule view.'
        ),
        'action_url': 'https://erp.university.edu/academics/timetable',
        'action_label': 'View Live Timetable',
    },
    'attendance': {
        'title': 'Attendance Compliance Policy',
        'response': (
            'Mandatory attendance criterion is 75%. If your attendance falls'
            ' between 60% and 74%, you can submit a medical condonation request'
            ' approved by your HOD and Dean office.'
        ),
        'action_url': 'https://erp.university.edu/academics/attendance',
        'action_label': 'Check Attendance Record',
    },
    'exam_schedule': {
        'title': 'Examination Cell',
        'response': (
            'Mid-term assessments begin in mid-October, while final practicals'
            ' and end-term theory exams are held in December. Hall tickets are'
            ' issued 3 days before exam commencement.'
        ),
        'action_url': 'https://erp.university.edu/exams',
        'action_label': 'Download Date Sheet',
    },
    'course_registration': {
        'title': 'Course Registration & Add/Drop',
        'response': (
            'To add or drop courses, navigate to Portal > Academics > Add/Drop'
            ' Courses during the active registration window. Seat availability'
            ' updates in real time.'
        ),
        'action_url': 'https://erp.university.edu/registration',
        'action_label': 'Manage Course Add/Drop',
    },
    'technical_issue': {
        'title': 'IT Helpdesk & Campus Wi-Fi',
        'response': (
            "Connect to 'UniSecure' using your student roll number credentials."
            ' For portal lockouts, password resets, or LMS issues, submit a'
            ' ticket directly to the campus IT cell.'
        ),
        'action_url': 'https://erp.university.edu/it-support',
        'action_label': 'Raise IT Ticket',
    },
    'financial_aid': {
        'title': 'Scholarships & Financial Aid',
        'response': (
            'Merit scholarships are available for students with a CGPA of 9.0'
            ' or above (up to 25% tuition fee waiver). Need-based aid and'
            ' national scholarship portals can be accessed online.'
        ),
        'action_url': 'https://erp.university.edu/scholarships',
        'action_label': 'View Scholarship Schemes',
    },
    'career_services': {
        'title': 'Training & Placement Cell (T&P)',
        'response': (
            'Most placement drives require a minimum 6.5 CGPA with zero active'
            ' backlogs. Resume verification, mock interviews, and drive'
            ' schedules are active on the placement portal.'
        ),
        'action_url': 'https://erp.university.edu/placements',
        'action_label': 'View Placement Drives',
    },
    'grades': {
        'title': 'Grades & Evaluation',
        'response': (
            'Semester grade cards and provisional mark sheets are available'
            ' under Results. Re-evaluation application windows remain open for 7'
            ' days post-result declaration.'
        ),
        'action_url': 'https://erp.university.edu/results',
        'action_label': 'Check Grade Cards',
    },
    'mental_health': {
        'title': 'Student Wellbeing & Counseling',
        'response': (
            'Free, confidential one-on-one counseling sessions and 24/7 mental'
            ' health helpline support are available. You do not have to handle'
            ' academic stress alone.'
        ),
        'action_url': 'https://erp.university.edu/wellness',
        'action_label': 'Book Confidential Session',
    },
    'academic_advising': {
        'title': 'Academic Advising & Mentorship',
        'response': (
            'Every student is assigned a dedicated faculty advisor. For branch'
            ' change petitions (minimum 8.0+ CGPA required after year 1), credit'
            ' transfers, or minor degrees, meet your advisor.'
        ),
        'action_url': 'https://erp.university.edu/advising',
        'action_label': 'Contact Faculty Advisor',
    },
    'housing': {
        'title': 'Hostel & Residential Services',
        'response': (
            'Hostel room allotments, maintenance requests, mess rebate forms,'
            " and room-change applications are handled through the Chief"
            " Warden's Office."
        ),
        'action_url': 'https://erp.university.edu/hostel',
        'action_label': 'Hostel Services Portal',
    },
    'student_life': {
        'title': 'Campus Life & Student Clubs',
        'response': (
            'Explore technical societies, coding clubs, cultural fests, and'
            ' sports team trials. You can also submit a new student club'
            ' proposal with at least 10 members.'
        ),
        'action_url': 'https://erp.university.edu/student-life',
        'action_label': 'Explore Clubs & Events',
    },
    'course_information': {
        'title': 'Course Information & Syllabi',
        'response': (
            'Detailed course syllabi, recommended textbooks, prerequisites, and'
            ' credit allocation for core and elective subjects can be downloaded'
            ' directly from the LMS.'
        ),
        'action_url': 'https://erp.university.edu/curriculum',
        'action_label': 'View Subject Syllabi',
    },
    'administration': {
        'title': 'Registrar & Student Services',
        'response': (
            'Apply online for bonafide certificates, transfer/migration'
            ' certificates, duplicate ID cards, and official transcripts via the'
            ' Student Self-Service counter.'
        ),
        'action_url': 'https://erp.university.edu/admin',
        'action_label': 'Request Official Document',
    },
    'campus_facilities': {
        'title': 'Campus Facilities & Amenities',
        'response': (
            'The Central Library is open from 09:00 AM to 08:00 PM on weekdays.'
            ' The sports arena, campus gym, medical clinic, and stationery shop'
            ' operate near the student activity block.'
        ),
        'action_url': 'https://erp.university.edu/facilities',
        'action_label': 'Campus Amenities Guide',
    },
    'safety': {
        'title': 'Campus Safety & Emergency',
        'response': (
            'For urgent security or medical assistance, contact the 24/7 campus'
            ' emergency control room or security office. Ambulances and first'
            ' aid are available on campus around the clock.'
        ),
        'action_url': 'https://erp.university.edu/safety',
        'action_label': 'Emergency Contact Directory',
    },
    'academic_calendar': {
        'title': 'Academic Calendar',
        'response': (
            'The academic calendar highlights semester start dates, mid-term'
            ' evaluations, semester break windows, and gazetted university'
            ' holidays.'
        ),
        'action_url': 'https://erp.university.edu/calendar',
        'action_label': 'View Academic Calendar',
    },
    'library': {
        'title': 'Library Services',
        'response': (
            'Borrow book limits, return renewals, thesis repositories, and'
            ' access to IEEE/Springer digital e-journals are managed via the'
            ' library portal.'
        ),
        'action_url': 'https://erp.university.edu/library',
        'action_label': 'Access Digital Library',
    },
    'academic_policies': {
        'title': 'Academic Policies & Regulations',
        'response': (
            'Official rules regarding assignment submission extensions,'
            ' examination grading scales, and academic integrity policies are'
            ' published by the Academic Council.'
        ),
        'action_url': 'https://erp.university.edu/policies',
        'action_label': 'Read Academic Policies',
    },
    'accessibility': {
        'title': 'Accessibility Services',
        'response': (
            'Special accommodations for examinations, assistive classroom'
            ' equipment, and ramp/elevator passes are coordinated through the'
            ' Accessibility Office.'
        ),
        'action_url': 'https://erp.university.edu/accessibility',
        'action_label': 'Request Accommodation',
    },
    'academic_support': {
        'title': 'Academic Support & Writing Center',
        'response': (
            'Writing centers and subject tutoring groups provide guidance on'
            ' APA/IEEE referencing, technical writing, and calculus assistance.'
        ),
        'action_url': 'https://erp.university.edu/academic-support',
        'action_label': 'Book Tutoring Session',
    },
    'student_services': {
        'title': 'Student Support Services',
        'response': (
            'Get support with enrollment verification, student ID replacement,'
            ' and official inquiries through student services.'
        ),
        'action_url': 'https://erp.university.edu/student-services',
        'action_label': 'Student Services Desk',
    },
    'research_support': {
        'title': 'Research & Innovation Support',
        'response': (
            'Guidance on research grants, laboratory safety protocols, paper'
            ' publications, and patent filings can be obtained via the R&D'
            ' cell.'
        ),
        'action_url': 'https://erp.university.edu/research',
        'action_label': 'Visit R&D Portal',
    },
    'academic_programs': {
        'title': 'Academic Programs & Honors',
        'response': (
            'Information on dual degrees, minor specializations, and honors'
            ' curriculum tracks is available under academic programs.'
        ),
        'action_url': 'https://erp.university.edu/programs',
        'action_label': 'Explore Honors & Minors',
    },
    'health': {
        'title': 'Health & Medical Services',
        'response': (
            'The campus health dispensary operates 24/7 with emergency first'
            ' aid, routine medical checkups, and ambulance availability.'
        ),
        'action_url': 'https://erp.university.edu/health',
        'action_label': 'Campus Dispensary Info',
    },
    'student_employment': {
        'title': 'On-Campus Student Employment',
        'response': (
            'Explore student assistantships, work-study openings, and research'
            ' assistant vacancies across university departments.'
        ),
        'action_url': 'https://erp.university.edu/employment',
        'action_label': 'View On-Campus Jobs',
    },
    'study_abroad': {
        'title': 'Study Abroad & Exchange Programs',
        'response': (
            'Learn about semester exchange agreements with partner universities'
            ' and international student credit transfer procedures.'
        ),
        'action_url': 'https://erp.university.edu/study-abroad',
        'action_label': 'Exchange Programs Guide',
    },
    'admissions': {
        'title': 'Admissions Desk',
        'response': (
            'Check lateral entry requirements, seat reservation criteria,'
            ' document verification dates, and counseling rounds.'
        ),
        'action_url': 'https://erp.university.edu/admissions',
        'action_label': 'Admissions Portal',
    },
}

FALLBACK_RESPONSE = {
    'title': 'Student Helpdesk',
    'response': (
        'I could not locate an exact match for your inquiry. Please consult'
        ' your department coordinator or visit the central administrative'
        ' office.'
    ),
    'action_url': 'mailto:support@university.edu',
    'action_label': 'Email Student Helpdesk',
}


# ==========================================
# 4. Smart Recommendation Prompts
# ==========================================
RECOMMENDATIONS = [
    {
        'label': '💰 B.Tech Annual Fee Structure',
        'query': 'What is the B.Tech fee structure and payment mode?',
    },
    {
        'label': '📅 Monday Class Timetable',
        'query': 'Show me my timetable and lecture timing for Monday',
    },
    {
        'label': '⚠️ 65% Attendance Rule',
        'query': 'I have 65% attendance, can I sit for exams?',
    },
    {
        'label': '📝 Midterm Exam Date Sheet',
        'query': 'When are the midterm exams scheduled?',
    },
    {
        'label': '💼 Placement & CGPA Eligibility',
        'query': 'Can I sit for placements if I have a backlog?',
    },
    {
        'label': '📶 Campus Wi-Fi Connection',
        'query': 'How do I connect to the university Wi-Fi network?',
    },
]


# ==========================================
# 5. Session State Management
# ==========================================
if 'chat_history' not in st.session_state:
  st.session_state.chat_history = []
if 'pending_query' not in st.session_state:
  st.session_state.pending_query = None


# ==========================================
# 6. Sidebar: Recommendations & Controls
# ==========================================
with st.sidebar:
  st.markdown('### 💡 Recommended Inquiries')
  st.caption('Click any recommended question to query the assistant:')

  for item in RECOMMENDATIONS:
    if st.button(item['label'], key=f"sidebar_rec_{item['label']}"):
      st.session_state.pending_query = item['query']

  st.markdown('---')
  if st.button('🗑️ Clear Chat History', use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.pending_query = None
    st.rerun()

  st.markdown('---')
  st.caption('🚀 Ready for Streamlit Cloud deployment.')


# ==========================================
# 7. Main Dashboard Area
# ==========================================
st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">🎓 University Student AI Dashboard</div>
    <div class="hero-subtitle">
        Instant guidance on semester tuition, lecture timetables, exam date sheets, attendance policies, and campus services.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Render Chat History
for msg in st.session_state.chat_history:
  with st.chat_message(msg['role']):
    st.markdown(msg['content'])
    if msg.get('metadata'):
      meta = msg['metadata']
      st.markdown(
          f'<a href="{meta["action_url"]}" target="_blank"'
          f' class="resource-badge">🔗 {meta["action_label"]}</a>',
          unsafe_allow_html=True,
      )
      st.caption(f"Topic: `{meta['intent']}` • Confidence: {meta['confidence']}")


# ==========================================
# 8. Query Processing & Inference
# ==========================================
chat_input = st.chat_input(
    'Type your question here (e.g., fee structure, exam dates, timetable)...'
)
query_to_process = chat_input or st.session_state.pending_query

if query_to_process:
  st.session_state.pending_query = None

  # Render User Message
  with st.chat_message('user'):
    st.markdown(query_to_process)
  st.session_state.chat_history.append(
      {'role': 'user', 'content': query_to_process}
  )

  # ML Inference
  vec = vectorizer.transform([query_to_process])
  pred_idx = classifier.predict(vec)[0]
  predicted_intent = label_encoder.inverse_transform([pred_idx])[0]

  # Confidence Calculation
  if hasattr(classifier, 'predict_proba'):
    probs = classifier.predict_proba(vec)[0]
    confidence = float(np.max(probs))
  elif hasattr(classifier, 'decision_function'):
    decision = classifier.decision_function(vec)
    confidence = float(1 / (1 + np.exp(-np.max(decision))))
  else:
    confidence = 1.0

  # Match response from merged Knowledge Base
  matched_entry = KNOWLEDGE_BASE.get(predicted_intent, FALLBACK_RESPONSE)
  formatted_reply = (
      f"**{matched_entry['title']}**\n\n{matched_entry['response']}"
  )

  # Render Bot Message
  with st.chat_message('assistant'):
    st.markdown(formatted_reply)
    st.markdown(
        f'<a href="{matched_entry["action_url"]}" target="_blank"'
        f' class="resource-badge">🔗 {matched_entry["action_label"]}</a>',
        unsafe_allow_html=True,
    )
    st.caption(
        f'Topic: `{predicted_intent}` • Confidence: {confidence * 100:.1f}%'
    )

  # Persist in Session History
  st.session_state.chat_history.append({
      'role': 'assistant',
      'content': formatted_reply,
      'metadata': {
          'intent': predicted_intent,
          'confidence': f'{confidence * 100:.1f}%',
          'action_url': matched_entry['action_url'],
          'action_label': matched_entry['action_label'],
      },
  })