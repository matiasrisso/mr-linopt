import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from streamlit.secrets import Secrets
from streamlit.state.session_state import SessionState

########## To run the app type:     streamlit run linopt_small_scipy_v2.py



# OBJECTIVE FUNCITON ##############################################################
st.sidebar.subheader('Objective parameters')
objective_type = st.sidebar.radio('',['Max','Min'], key='objective_type')

# Variables
num_variab = st.sidebar.slider('Number of variables', min_value=2, max_value=20, value=2, key='number_variables')
# Update the default bounds for the new number of variables
prev_num_variab = len(st.session_state.bounds)
if prev_num_variab >= num_variab:
    st.session_state.bounds = st.session_state.bounds[0:num_variab]
else:
    for i in range(num_variab - prev_num_variab):
        st.session_state.bounds.append((0, np.inf))


X_n = ['x_'+str(i+1) for i in range(num_variab)] # names
C_of  = [1 for i in range(num_variab)]  # coefs for the objetive funciton
C_of_latex = C_of

for i in range(num_variab):
    C_of[i] = st.sidebar.number_input('X'+str(i+1)+' Coef', value=1.0, key='obj_coef_'+str(i+1))
if objective_type == 'Max':
    C_of = [-i for i in C_of] # reverse the coefs if it is a max objetive, solver only does min

# Show objective function
OF_string = ''
for i in range(num_variab):
    if i == 0:
        OF_string += ( str(C_of_latex[i]) + ' * ' + X_n[i] )
    elif C_of_latex[i] >= 0:
        OF_string += ( ' + ' + str(abs(C_of_latex[i])) + ' * ' + X_n[i] )
    else:
        OF_string += ( ' - ' + str(abs(C_of_latex[i])) + ' * ' + X_n[i] )

st.title('Linear Optimization')
st.header('Objetive Function')
st.latex(objective_type + ' Z = ' + OF_string)



# RESTRICTIONS ##############################################################
if 'new_model' not in st.session_state: # Initiation of parameters
    st.session_state.new_model = True
    st.session_state.A_ub = []
    st.session_state.b_ub = []
    st.session_state.A_eq = []
    st.session_state.b_eq = []
    st.session_state.bounds = [(0, np.inf) for i in range(num_variab)]

st.header('Restrictions')
edit_restrictions = st.radio('Add or delete restrictions', ['Yes', 'No'])

if edit_restrictions == 'Yes':
    # Add restriction
    st.subheader('Add restrictions or bounds')
    with st.beta_container():
        col1, col2 = st.beta_columns(2)
        restriction_type = col1.radio('Type', ['X < U', 'L < X', 'X = B', 'Variable bounds'])

        if restriction_type == 'X < U':
            b_val = col2.number_input('Upper limit')
        elif restriction_type == 'L < X':
            b_val = - col2.number_input('Lower limit') # negative as it is an opposite upper bound
        elif restriction_type == 'X = B':
            b_val = col2.number_input('Equal value')
        
        if restriction_type != 'Variable bounds':
            A_vect = [i for i in range(num_variab)]
            for i in range(num_variab):
                if restriction_type != 'L < X':
                    A_vect[i] = st.number_input('X'+str(i+1)+' Coef', value=1.0, key='res_coef_'+str(i+1))
                else:
                    A_vect[i] = - st.number_input('X'+str(i+1)+' Coef', value=1.0, key='res_coef_'+str(i+1)) # negative as it is an opposite upper bound
        else:
            Bound_list = [[0, np.inf] for i in range(num_variab)]
            for i in range(num_variab):
                bound_col1, bound_col2, bound_col3, bound_col4 = st.beta_columns(4) # has to be in the loop to make the columns every time
                if bound_col1.radio('X'+str(i+1)+' min = -inf', ['Yes', 'No'], index=1) == 'Yes':
                    Bound_list[i][0] = -np.inf
                else:
                    Bound_list[i][0] = bound_col2.number_input('X'+str(i+1)+' min', value=0.0, key='min_bound_'+str(i+1))
                if bound_col3.radio('X'+str(i+1)+' max = inf', ['Yes', 'No']) == 'Yes':
                    Bound_list[i][1] = np.inf
                else:
                    Bound_list[i][1] = bound_col4.number_input('X'+str(i+1)+' max', value=10, key='max_bound_'+str(i+1))

        if st.button('Confirm'):
            # Add to the model
            if restriction_type == 'X < U' or restriction_type == 'L < X':
                st.session_state.A_ub.append(A_vect)
                st.session_state.b_ub.append(b_val)
            elif restriction_type == 'X = B':
                st.session_state.A_eq.append(A_vect)
                st.session_state.b_eq.append(b_val)
            else:
                Bound_tup = [tuple(Bound_list[i]) for i in range(num_variab)] # needs to be a list of tuples for the solver
                st.session_state.bounds = Bound_tup

    
    st.subheader('Delete restrictions')
    total_ineq = len(st.session_state.b_ub)
    total_eq = len(st.session_state.b_eq)
    total_restr = len(st.session_state.b_ub) + len(st.session_state.b_eq)

    container_del = st.empty()
    res_select_del = container_del.multiselect('Choose', [int(i+1) for i in range(total_restr)])

    st.session_state.cond_del = [] # vect of bool to choose what to keep, in state to keep after button
    for i in range(total_restr):
        if (i+1) not in res_select_del:
            st.session_state.cond_del.append(True)
        else:
            st.session_state.cond_del.append(False)

    if st.button('Delete'):
        # Only keep the unselected restrictions for ineq and eq type
        st.session_state.A_ub = [st.session_state.A_ub[i] for i in range(total_ineq) if st.session_state.cond_del[i]]
        st.session_state.b_ub = [st.session_state.b_ub[i] for i in range(total_ineq) if st.session_state.cond_del[i]]
        st.session_state.A_eq = [st.session_state.A_eq[i] for i in range(total_eq) if st.session_state.cond_del[i+total_ineq]]
        st.session_state.b_eq = [st.session_state.b_eq[i] for i in range(total_eq) if st.session_state.cond_del[i+total_ineq]]


# Show restrictions
if st.checkbox('Show restrictions', value=True):
    res_counter = 1 # to enumerate them when printed

    # Inequalities
    if len(st.session_state.A_ub) > 0:
        for i, restr in enumerate(st.session_state.A_ub):
            r_string = 'R_' + str(res_counter) + ': '
            for j in range(num_variab):
                if restr[j] != 0:
                    if j == 0:
                        r_string += ( str(restr[j]) + ' * ' + X_n[j] )
                    elif restr[j] >= 0:
                        r_string += ( ' + ' + str(abs(restr[j])) + ' * ' + X_n[j] )
                    else:
                        r_string += ( ' - ' + str(abs(restr[j])) + ' * ' + X_n[j] )

            r_string += ( ' \leq ' + (str(st.session_state.b_ub[i]) if st.session_state.b_ub[i] !=0 else '0.0') ) # not to get -0
            st.latex(r_string)
            res_counter += 1

    # Equalities
    if len(st.session_state.A_eq) > 0:
        for i, restr in enumerate(st.session_state.A_eq):
            r_string = 'R_' + str(res_counter) + ': '
            for j in range(num_variab):
                if restr[j] != 0:
                    if j == 0:
                        r_string += ( str(restr[j]) + ' * ' + X_n[j] )
                    elif restr[j] >= 0:
                        r_string += ( ' + ' + str(abs(restr[j])) + ' * ' + X_n[j] )
                    else:
                        r_string += ( ' - ' + str(abs(restr[j])) + ' * ' + X_n[j] )

            r_string += ( ' = ' + str(st.session_state.b_eq[i]))
            st.latex(r_string)
            res_counter += 1

    # Bounds
    if len(st.session_state.bounds) > 0:
        for i, bound in enumerate(st.session_state.bounds):
            L_b = str(bound[0]) if bound[0] != -np.inf else '- \infty'
            U_b = str(bound[1]) if bound[1] != np.inf else ' \infty'
            b_string = ( L_b + ' \leq ' + X_n[i] + ' \leq ' + U_b )
            st.latex(b_string)
    else:
        for i in range(num_variab):
            b_string = ( '0' + ' \leq ' + X_n[i] + ' \leq ' + ' \infty' )
            st.latex(b_string)



# Plot 2D system ################################################################################################
if num_variab == 2:
    fig = go.Figure()
    fig.add_vline(x=0)
    fig.add_hline(y=0)
    # Bounds only add if not infinity
    if st.session_state.bounds[0][0] != -np.inf:
        fig.add_vline(x=st.session_state.bounds[0][0],line_dash="dash", line_color="red")
    if st.session_state.bounds[0][1] != np.inf:
        fig.add_vline(x=st.session_state.bounds[0][1], line_dash="dash", line_color="red")
    if st.session_state.bounds[1][0] != -np.inf:
        fig.add_hline(y=st.session_state.bounds[1][0], line_dash="dash", line_color="blue")
    if st.session_state.bounds[1][1] != np.inf:
        fig.add_hline(y=st.session_state.bounds[1][1], line_dash="dash", line_color="blue")

    # if st.session_state.bounds[0][0] != -np.inf:
    #     fig.add_trace(go.Scatter(x=np.full(2,st.session_state.bounds[0][0]), y=[-50,50], fill='tonextx'))
    # if st.session_state.bounds[0][1] != np.inf:
    #     fig.add_trace(go.Scatter(x=np.full(2,st.session_state.bounds[0][1]), y=[-50,50], fill='tonextx'))
    # if st.session_state.bounds[1][0] != -np.inf:
    #     fig.add_trace(go.Scatter(x=[-50,50], y=np.full(2,st.session_state.bounds[1][0]), stackgroup='one'))
    # if st.session_state.bounds[1][1] != np.inf:
    #     fig.add_trace(go.Scatter(x=[-50,50], y=np.full(2,st.session_state.bounds[1][1]), stackgroup='one'))

    # Plot restriction lines
    res_counter = 1
    # Inequalities
    if len(st.session_state.A_ub) > 0:
        for A, b in zip(st.session_state.A_ub, st.session_state.b_ub):
            x = np.linspace(-50, 50, 1000)
            y = (b - A[0] * x) / A[1]
            name = 'R' + str(res_counter)
            fig.add_trace(go.Scatter(x=x, y=y, mode = 'lines', line_dash='dot',name=name))
            res_counter += 1
    # Equalities
    if len(st.session_state.A_eq) > 0:
        for A, b in zip(st.session_state.A_eq, st.session_state.b_eq):
            x = np.linspace(-50, 50, 1000)
            y = (b - A[0] * x) / A[1]
            name = 'R' + str(res_counter)
            fig.add_trace(go.Scatter(x=x, y=y, mode = 'lines', name=name))
            res_counter += 1

    fig.update_layout(width =800, height=700)
    st.plotly_chart(fig)

# SOLUTION ##################################################################################################

st.header('Solve the model')
if st.button('Solve'):
    model = {'C_of': C_of}
    for param in ['A_ub', 'b_ub', 'A_eq', 'b_eq', 'bounds']:
        if len(st.session_state[param]) > 0:
            model[param] = st.session_state[param] # in the funciton, no values have to be None and not empy lists
        else:
            model[param] = None

    st.session_state.result = linprog(model['C_of'], A_ub=model['A_ub'], b_ub=model['b_ub'], A_eq=model['A_eq'], b_eq=model['b_eq'], 
                    bounds=model['bounds'], method='simplex')

    if st.session_state.result['success']:
        st.balloons()


if 'result' in st.session_state:
    result = st.session_state.result

    st.subheader('Status')
    st.write(result['message'])

    st.subheader('Variables')
    st.write( pd.DataFrame(result['x'], columns=['Optimal values'], index=X_n)) # X values
    objetive_result = result['fun'] if objective_type == 'Min' else -result['fun']
    st.subheader('Objective function = ' + str(objetive_result))




