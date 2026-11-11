"""iMEWS+ | Phase 2 surrogate dashboard. Run: python -m streamlit run app.py"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from schema import YEARS_TRAINED, CONTINUOUS_GROUPS, CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, PHASE2_COLUMNS, display_label
from phase2 import load_phase2_model, predict_phase2, what_if, snapshot, EXPENSE_TYPES

st.set_page_config(page_title='iMEWS+ · Business outlook', page_icon='📈', layout='wide')
st.markdown('''<style>
[data-testid="stAppViewContainer"]{background:#f5f7fa}
.block-container{max-width:1190px;padding-top:2rem}
[data-testid="stMetric"]{background:#fff;border:1px solid #e4eaf1;border-radius:12px;padding:1rem}
h1,h2,h3{color:#17314c}
.hero{background:linear-gradient(110deg,#102a43,#20566a);color:#fff;border-radius:18px;padding:1.7rem 2rem;margin-bottom:1rem}
.hero h1{color:#fff;margin:0}.hero p{color:#d5e8ed;margin:.3rem 0 0}
.small{color:#50657a;font-size:.88rem}
</style>''', unsafe_allow_html=True)

MODEL_PATH = Path(__file__).resolve().parent / 'model' / 'phase2_surrogate_model.joblib'

@st.cache_resource(show_spinner=False)
def get_model(path: str, modified: float | None):
    return load_phase2_model(path)

try:
    artifact = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None)
    model_error = None
except Exception as exc:
    artifact = None
    model_error = str(exc)

st.markdown('<div class="hero"><h1>iMEWS+</h1><p>Intelligent Business Early Warning System · Your business outlook, made understandable.</p></div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown('## iMEWS+')
    st.write('**1.** Business information')
    st.write('**2.** Health dashboard')
    st.write('**3.** What-if analysis')
    st.divider()
    if artifact is not None:
        st.success('Phase 2 model connected')
    else:
        st.warning('Phase 2 model not found' if not model_error else 'Phase 2 model could not be loaded')
    st.caption('The model uses 75 numeric fields and 29 binary indicators over a four-year historical window.')
    st.caption('Inputs are processed locally by the running app. Do not use an untrusted .joblib file.')

if artifact is None:
    st.warning('Predictions are disabled. Place the original **phase2_surrogate_model.joblib** from your notebook output in **iMEWS_plus/model/** and restart the app. No alternate model or invented probability will be used.')
    if model_error:
        st.error('Model load error: ' + model_error)

st.subheader('1 · Business information')
st.write('Enter all variables used to train the Phase 2 model. Use four consecutive years of business records, with the most recent year last.')
RELATIVE_YEARS = ('4 years before prediction', '3 years before prediction', '2 years before prediction', '1 year before prediction')
name = st.text_input('Business name', value=st.session_state.get('business_name','My business'))
currency = st.selectbox('Currency (display label only)', ['LKR','USD','CAD','EUR','GBP','Other'])
st.caption('Enter actual values. Use 0 only if the true value is zero. Blank fields must be completed. The app does not convert currencies.')

ANNUAL_GROUPS = {
    'People & staffing': ['Owners','Admin_Staff','Tech_Staff'],
    'Annual operating costs': EXPENSE_TYPES,
    'Business revenue & other income': ['Gross_annual_revenue','Other_income_total'],
    'Investment, cash inflows & debt': ['Cash_inflow_investing','Cash_inflow_financial_activities','total_investment','non-performing_debt'],
}
OWNER_FIELDS = CONTINUOUS_GROUPS['Owner resources']
SPECIAL_FIELDS = ['Minimum_2023_Monthly_profit','Maximum_2023_Monthly_profit']
def relative_feature_label(column: str) -> str:
    for year, period in zip(YEARS_TRAINED, RELATIVE_YEARS):
        if column.endswith(f'_{year}'):
            return display_label(column[:-5]) + f' ({period})'
    return display_label(column)

CAT_GROUPS = {
    'Business owner and primary income': CATEGORICAL_COLUMNS[:7],
    'District & raw materials': CATEGORICAL_COLUMNS[7:14],
    'Banking & records': CATEGORICAL_COLUMNS[14:22],
    'Loan sources': CATEGORICAL_COLUMNS[22:],
}

# The data editors show one familiar financial table per category instead of 68 separate widgets.
def make_table(prefixes, key):
    columns = [display_label(p) for p in prefixes]
    default = pd.DataFrame({ 'Period': list(RELATIVE_YEARS), **{label: [None]*4 for label in columns}})
    if key not in st.session_state:
        st.session_state[key] = default
    config = {'Period': st.column_config.TextColumn('Period', disabled=True, width='medium')}
    for label in columns:
        config[label] = st.column_config.NumberColumn(label, min_value=0, step=1 if label in ('Owners','Admin staff','Tech staff') else 100, format='%.2f')
    return st.data_editor(st.session_state[key], hide_index=True, use_container_width=True, num_rows='fixed', column_config=config, key=f'editor_{key}')

with st.expander('A. Owner information', expanded=True):
    owner_inputs = {}
    cols = st.columns(2)
    for i, field in enumerate(OWNER_FIELDS):
        with cols[i%2]:
            owner_inputs[field] = st.number_input(display_label(field), value=None, step=1.0, key=f'field_{field}', help='Required; provide the actual value. Use 0 only when applicable.')
    st.caption('“Revenue percent withdrawn” should use the same units as the model training data; verify whether your dataset stores percentages as 0–100 or 0–1.')

annual_tables = {}
for heading, prefixes in ANNUAL_GROUPS.items():
    with st.expander('B. ' + heading, expanded=heading=='Business revenue & other income'):
        st.caption('Each row represents a year relative to the prediction date, from four years before to one year before.')
        annual_tables[heading] = (prefixes, make_table(prefixes, 'table_'+heading.replace(' ','_')))

with st.expander('C. Lowest and highest monthly profit (1 year before prediction)', expanded=False):
    st.caption('Enter the lowest and highest monthly profit from the most recent year. Profit may be negative.')
    special_inputs = {}
    for field in SPECIAL_FIELDS:
        special_inputs[field] = st.number_input('Minimum monthly profit' if field.startswith('Minimum') else 'Maximum monthly profit', value=None, step=100.0, key=f'field_{field}')

with st.expander('D. Business details and financial practices', expanded=False):
    st.caption('Select Yes (1) or No (0) for each indicator. The recorded values are passed to the model without changing its 0/1 coding. Confirm the original dataset’s encoding, particularly owner gender.')
    cat_inputs = {}
    for heading, fields in CAT_GROUPS.items():
        st.markdown('**'+heading+'**')
        a,b = st.columns(2)
        for i, field in enumerate(fields):
            with (a if i%2==0 else b):
                answer = st.selectbox(display_label(field), ['Select an answer','No · 0','Yes · 1'], key=f'cat_{field}')
                cat_inputs[field] = None if answer=='Select an answer' else (1 if answer=='Yes · 1' else 0)

values = {}
values.update(owner_inputs)
values.update(special_inputs)
values.update(cat_inputs)
for _, (prefixes, table) in annual_tables.items():
    for row_num, training_year in enumerate(YEARS_TRAINED):
        for prefix in prefixes:
            cell = table.iloc[row_num][display_label(prefix)]
            values[f'{prefix}_{training_year}'] = None if pd.isna(cell) else float(cell)

missing = [c for c in PHASE2_COLUMNS if values.get(c) is None or pd.isna(values.get(c))]
st.caption(f'Input completeness: {len(PHASE2_COLUMNS)-len(missing)} / {len(PHASE2_COLUMNS)} model variables filled')
if st.button('Analyse my business', type='primary', use_container_width=True):
    if missing:
        st.error(f'Please complete the remaining {len(missing)} model fields. First missing fields: '+', '.join(relative_feature_label(c) for c in missing[:8]))
    else:
        st.session_state.baseline = dict(values)
        st.session_state.business_name = name
        st.session_state.business_currency = currency
        st.success('Information saved for this session. Dashboard updated below.')

if 'baseline' not in st.session_state:
    st.info('Complete the information above and click **Analyse my business** to see your dashboard.')
    st.stop()

baseline = st.session_state.baseline
label = st.session_state.get('business_currency','LKR')
base_stats = snapshot(baseline)
st.divider()
st.subheader(f'2 · {st.session_state.business_name} — business outlook')
st.caption('Assessment for the **beginning of the next year** using the preceding four years of business records. The trained surrogate has not been validated across every possible calendar period.')

base_prediction = None
if artifact is not None:
    try:
        base_prediction = predict_phase2(baseline, artifact)
    except Exception as exc:
        st.error('Prediction failed: '+str(exc))

if base_prediction:
    a,b,c = st.columns(3)
    a.metric('Estimated probability of staying active', f"{base_prediction['active']:.1%}")
    b.metric('Estimated probability of closure', f"{base_prediction['closed']:.1%}")
    c.metric('Model risk classification', 'Higher risk' if base_prediction['higher_risk'] else 'Lower risk')
    st.caption(f"Classification uses the saved Phase 1 closure threshold: {base_prediction['threshold']:.1%}. Phase 2 predicts the **Phase 1 estimated probability of remaining active**; it is not a direct survival classifier or an independently validated future forecast.")
else:
    st.info('Financial indicators are shown below; model probabilities require the original saved Phase 2 file.')

r1,r2,r3,r4 = st.columns(4)
r1.metric('Most recent year annual revenue', f"{base_stats['revenue']:,.0f} {label}")
r2.metric('Reported operating costs', f"{base_stats['costs']:,.0f} {label}")
r3.metric('Revenue minus listed costs', f"{base_stats['balance']:,.0f} {label}")
r4.metric('Non-performing debt', f"{base_stats['debt']:,.0f} {label}")
st.caption('Revenue minus listed costs is a simplified operating indicator, **not** accounting net profit; other income, tax, and accounting adjustments may not be included.')

annual_revenue = [baseline[f'Gross_annual_revenue_{y}'] for y in YEARS_TRAINED]
annual_costs = [sum(baseline[f'{p}_{y}'] for p in EXPENSE_TYPES) for y in YEARS_TRAINED]
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(RELATIVE_YEARS),y=annual_revenue,mode='lines+markers',name='Revenue',line={'color':'#078c81','width':3}))
fig.add_trace(go.Scatter(x=list(RELATIVE_YEARS),y=annual_costs,mode='lines+markers',name='Listed operating costs',line={'color':'#df9144','width':3}))
fig.update_layout(height=290,margin=dict(l=0,r=0,t=15,b=0),xaxis=dict(type='category',title='Period'),yaxis_title=label,legend=dict(orientation='h',y=1.15),plot_bgcolor='white',paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig,use_container_width=True)
notes=[]
if base_stats['balance']<0: notes.append('Reported operating costs exceed gross annual revenue.')
if base_stats['growth'] is not None and base_stats['growth']<0:notes.append('Revenue declined relative to the previous year.')
if base_stats['debt']>0: notes.append('The business reports non-performing debt; monitor repayments and liquidity.')
if notes:
    for n in notes: st.warning(n)
else: st.success('No simple warning sign detected in the three headline financial indicators. This does not guarantee business survival.')

st.divider()
st.subheader('3 · What-if analysis')
st.write('Explore changes to **the most recent year (1 year before prediction) only**. Choose a revenue change, a proportional change across the eight reported operating-cost categories, and a change to non-performing debt. Other model inputs remain unchanged.')
with st.container(border=True):
    c1,c2,c3 = st.columns(3)
    rev_change = c1.slider('Revenue change (%)',-50,100,0,5)
    cost_change = c2.slider('Operating-cost change (%)',-50,100,0,5)
    debt_change = c3.slider('Non-performing debt change (%)',-100,100,0,5)
    if baseline['non-performing_debt_2023'] == 0:
        st.caption('Current non-performing debt is zero, so percentage changes leave debt at zero.')
scenario_values = what_if(baseline,rev_change,cost_change,debt_change)
new_stats = snapshot(scenario_values)
a,b,c = st.columns(3)
a.metric('Scenario revenue',f"{new_stats['revenue']:,.0f} {label}",delta=f"{new_stats['revenue']-base_stats['revenue']:,.0f}")
b.metric('Scenario listed costs',f"{new_stats['costs']:,.0f} {label}",delta=f"{new_stats['costs']-base_stats['costs']:,.0f}",delta_color='inverse')
c.metric('Scenario revenue less costs',f"{new_stats['balance']:,.0f} {label}",delta=f"{new_stats['balance']-base_stats['balance']:,.0f}")
if artifact is not None and base_prediction is not None:
    try:
        simulated = predict_phase2(scenario_values,artifact)
        a,b,c = st.columns(3)
        a.metric('Current estimated closure probability',f"{base_prediction['closed']:.1%}")
        b.metric('Scenario estimated closure probability',f"{simulated['closed']:.1%}",delta=f"{(simulated['closed']-base_prediction['closed'])*100:+.1f} pp",delta_color='inverse')
        c.metric('Scenario classification','Higher risk' if simulated['higher_risk'] else 'Lower risk')
        if simulated['higher_risk'] != base_prediction['higher_risk']:
            st.info('The model classification changes under this scenario.')
        else:
            st.caption('The model classification remains the same under this scenario.')
    except Exception as exc:
        st.error('What-if model calculation failed: '+str(exc))
st.caption('What-if values are hypothetical model sensitivities, not causal forecasts. Increasing revenue may also affect costs, staff, and cash flow in practice; the scenario holds those variables fixed unless directly adjusted.')

export_df = pd.DataFrame([{'model_variable':c,'value':baseline[c]} for c in PHASE2_COLUMNS])
st.download_button('Download my model inputs (CSV)',export_df.to_csv(index=False).encode('utf-8'),file_name='imews_phase2_inputs.csv',mime='text/csv')
