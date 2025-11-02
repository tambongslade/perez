from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import markdown2
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request models
class ChatRequest(BaseModel):
    message: str
    session_data: dict = None

app = FastAPI(title="Structural Analysis Visualizer")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


def process_excel_file(file_content: bytes):
    """Process the Excel file and extract datasets"""
    try:
        df = pd.read_excel(BytesIO(file_content), sheet_name=0)

        # Extract dataset 1 (Reference case)
        dataset1 = df.iloc[1:, [0, 1]].copy()
        dataset1.columns = ['U_mm', 'F_kn']
        dataset1 = dataset1.dropna()
        dataset1['U_mm'] = pd.to_numeric(dataset1['U_mm'], errors='coerce')
        dataset1['F_kn'] = pd.to_numeric(dataset1['F_kn'], errors='coerce')
        dataset1 = dataset1.dropna()

        # Extract dataset 2 (Test data)
        dataset2 = df.iloc[1:, [4, 5]].copy()
        dataset2.columns = ['u', 'RF']
        dataset2 = dataset2.dropna()
        dataset2['u'] = pd.to_numeric(dataset2['u'], errors='coerce')
        dataset2['RF'] = pd.to_numeric(dataset2['RF'], errors='coerce')
        dataset2 = dataset2.dropna()

        # Get test name from header
        test_name = df.iloc[0, 4] if pd.notna(df.iloc[0, 4]) else "Test Data"

        return {
            'dataset1': dataset1,
            'dataset2': dataset2,
            'test_name': str(test_name)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


def extract_envelope_curve(displacement, force):
    """Extract smooth envelope curve without anomalies from hysteresis data"""
    import numpy as np
    from scipy import interpolate

    # Convert to numpy arrays for easier manipulation
    disp = np.array(displacement)
    force_vals = np.array(force)

    # Create displacement bins for envelope extraction
    num_bins = 50
    disp_bins = np.linspace(disp.min(), disp.max(), num_bins)
    envelope_disp = []
    envelope_force = []

    for i, target_disp in enumerate(disp_bins):
        # Find points within a range around target displacement
        tolerance = (disp.max() - disp.min()) / (num_bins * 2)
        mask = np.abs(disp - target_disp) <= tolerance

        if np.any(mask):
            local_forces = force_vals[mask]

            # Take the maximum absolute force (peak envelope)
            max_abs_idx = np.argmax(np.abs(local_forces))
            peak_force = local_forces[max_abs_idx]

            envelope_disp.append(target_disp)
            envelope_force.append(peak_force)

    if len(envelope_disp) < 3:
        return {'displacement': [], 'force': []}

    # Convert to numpy arrays
    envelope_disp = np.array(envelope_disp)
    envelope_force = np.array(envelope_force)

    # Sort by displacement
    sort_idx = np.argsort(envelope_disp)
    envelope_disp = envelope_disp[sort_idx]
    envelope_force = envelope_force[sort_idx]

    # Apply smoothing using a moving average to remove anomalies
    window_size = max(3, len(envelope_force) // 10)
    if window_size % 2 == 0:
        window_size += 1  # Make sure window size is odd

    smoothed_force = np.convolve(envelope_force, np.ones(window_size)/window_size, mode='same')

    # Use spline interpolation for additional smoothness
    try:
        # Create more points for smoother curve
        dense_disp = np.linspace(envelope_disp.min(), envelope_disp.max(), 100)
        spline = interpolate.UnivariateSpline(envelope_disp, smoothed_force, s=len(envelope_disp)*0.1)
        dense_force = spline(dense_disp)

        return {
            'displacement': dense_disp.tolist(),
            'force': dense_force.tolist()
        }
    except:
        # Fallback to smoothed data if spline fails
        return {
            'displacement': envelope_disp.tolist(),
            'force': smoothed_force.tolist()
        }


def extract_key_envelope_points(displacement, force):
    """Extract envelope curve at key displacement points: -40, -20, 0, 20, 40"""
    import numpy as np

    # Convert to numpy arrays
    disp = np.array(displacement)
    force_vals = np.array(force)

    # Define key displacement points
    key_points = np.array([-40, -20, 0, 20, 40])

    # Filter key points to be within data range
    key_points = key_points[
        (key_points >= disp.min()) & (key_points <= disp.max())
    ]

    if len(key_points) < 2:
        return {'displacement': [], 'force': []}

    # Extract peak forces at key displacement points
    envelope_disp = []
    envelope_force = []

    for target_disp in key_points:
        # Find points within a range around target displacement
        tolerance = 5  # 5mm tolerance around each key point
        mask = np.abs(disp - target_disp) <= tolerance

        if np.any(mask):
            local_forces = force_vals[mask]

            # Take the maximum absolute force (peak envelope)
            max_abs_idx = np.argmax(np.abs(local_forces))
            peak_force = local_forces[max_abs_idx]

            envelope_disp.append(target_disp)
            envelope_force.append(peak_force)

    return {
        'displacement': envelope_disp,
        'force': envelope_force
    }


def create_plots(data):
    """Generate all plots"""
    dataset1 = data['dataset1']
    dataset2 = data['dataset2']
    test_name = data['test_name']

    plots = {}
    
    # Generate envelope curves first for use in multiple plots
    envelope_ref = extract_envelope_curve(dataset1['U_mm'], dataset1['F_kn'])
    envelope_test = extract_envelope_curve(dataset2['u'], dataset2['RF'])

    # 1. Hysteresis curve - Reference case
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=dataset1['U_mm'],
        y=dataset1['F_kn'],
        mode='lines+markers',
        name='Cas de Référence',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    fig1.update_layout(
        title='Cas de Référence : Courbe Force-Déplacement',
        xaxis_title='Déplacement U (mm)',
        yaxis_title='Force F (kN)',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    plots['hysteresis_ref'] = json.loads(fig1.to_json())

    # 2. Hysteresis curve - Test data
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=dataset2['u'],
        y=dataset2['RF'],
        mode='lines+markers',
        name=test_name,
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ))
    fig2.update_layout(
        title=f'{test_name} : Courbe Force-Déplacement',
        xaxis_title='Déplacement u (mm)',
        yaxis_title='Force de Réaction RF (kN)',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    plots['hysteresis_test'] = json.loads(fig2.to_json())

    # 3. Comparison plot
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=dataset1['U_mm'],
        y=dataset1['F_kn'],
        mode='lines',
        name='Cas de Référence',
        line=dict(color='blue', width=2)
    ))
    fig3.add_trace(go.Scatter(
        x=dataset2['u'],
        y=dataset2['RF'],
        mode='lines',
        name=test_name,
        line=dict(color='red', width=2)
    ))
    fig3.update_layout(
        title='Comparaison : Référence vs Données de Test',
        xaxis_title='Déplacement (mm)',
        yaxis_title='Force (kN)',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    plots['comparison'] = json.loads(fig3.to_json())

    # 4. Displacement vs Index (loading history)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        y=dataset1['U_mm'],
        mode='lines',
        name='Cas de Référence',
        line=dict(color='blue', width=2)
    ))
    fig4.add_trace(go.Scatter(
        y=dataset2['u'],
        mode='lines',
        name=test_name,
        line=dict(color='red', width=2)
    ))
    fig4.update_layout(
        title='Historique de Chargement : Déplacement vs Pas de Temps',
        xaxis_title='Pas de Temps',
        yaxis_title='Déplacement (mm)',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    plots['loading_history'] = json.loads(fig4.to_json())

    # 5. Force vs Index
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        y=dataset1['F_kn'],
        mode='lines',
        name='Cas de Référence',
        line=dict(color='blue', width=2)
    ))
    fig5.add_trace(go.Scatter(
        y=dataset2['RF'],
        mode='lines',
        name=test_name,
        line=dict(color='red', width=2)
    ))
    fig5.update_layout(
        title='Historique de Force : Force vs Pas de Temps',
        xaxis_title='Pas de Temps',
        yaxis_title='Force (kN)',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    plots['force_history'] = json.loads(fig5.to_json())

    # 6. Energy dissipation (area under hysteresis loop approximation)
    # Calculate cumulative energy for each dataset
    dataset1_sorted = dataset1.sort_values('U_mm')
    dataset2_sorted = dataset2.sort_values('u')

    energy1 = []
    energy2 = []

    for i in range(1, len(dataset1)):
        area = abs(dataset1['F_kn'].iloc[i] * (dataset1['U_mm'].iloc[i] - dataset1['U_mm'].iloc[i-1]))
        energy1.append(sum([abs(dataset1['F_kn'].iloc[j] * (dataset1['U_mm'].iloc[j] - dataset1['U_mm'].iloc[j-1])) for j in range(1, i+1)]))

    for i in range(1, len(dataset2)):
        area = abs(dataset2['RF'].iloc[i] * (dataset2['u'].iloc[i] - dataset2['u'].iloc[i-1]))
        energy2.append(sum([abs(dataset2['RF'].iloc[j] * (dataset2['u'].iloc[j] - dataset2['u'].iloc[j-1])) for j in range(1, i+1)]))

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=list(range(1, len(energy1)+1)),
        y=energy1,
        mode='lines',
        name='Cas de Référence',
        line=dict(color='blue', width=2)
    ))
    fig6.add_trace(go.Scatter(
        x=list(range(1, len(energy2)+1)),
        y=energy2,
        mode='lines',
        name=test_name,
        line=dict(color='red', width=2)
    ))
    fig6.update_layout(
        title='Dissipation d\'Énergie Cumulative',
        xaxis_title='Étape',
        yaxis_title='Énergie Cumulative (kN·mm)',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    plots['energy_dissipation'] = json.loads(fig6.to_json())

    # 7. Reference Case Envelope Curve
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=envelope_ref['displacement'],
        y=envelope_ref['force'],
        mode='lines+markers',
        name='Enveloppe Référence',
        line=dict(color='darkblue', width=4),
        marker=dict(size=6, color='darkblue')
    ))
    fig7.update_layout(
        title='Cas de Référence : Courbe Enveloppe Unifiée (Squelette)',
        xaxis_title='Déplacement U (mm)',
        yaxis_title='Force F (kN)',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    plots['envelope_ref'] = json.loads(fig7.to_json())

    # 8. Test Data Envelope Curve
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(
        x=envelope_test['displacement'],
        y=envelope_test['force'],
        mode='lines+markers',
        name=f'Enveloppe {test_name}',
        line=dict(color='darkred', width=4),
        marker=dict(size=6, color='darkred')
    ))
    fig8.update_layout(
        title=f'{test_name} : Courbe Enveloppe Unifiée (Squelette)',
        xaxis_title='Déplacement u (mm)',
        yaxis_title='Force de Réaction RF (kN)',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    plots['envelope_test'] = json.loads(fig8.to_json())

    # 9. Envelope Curves Comparison
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(
        x=envelope_ref['displacement'],
        y=envelope_ref['force'],
        mode='lines+markers',
        name='Enveloppe Référence',
        line=dict(color='blue', width=4),
        marker=dict(size=6, color='blue')
    ))
    fig9.add_trace(go.Scatter(
        x=envelope_test['displacement'],
        y=envelope_test['force'],
        mode='lines+markers',
        name=f'Enveloppe {test_name}',
        line=dict(color='red', width=4),
        marker=dict(size=6, color='red')
    ))
    fig9.update_layout(
        title='Comparaison des Courbes Enveloppes Unifiées : Courbes Squelettes',
        xaxis_title='Déplacement (mm)',
        yaxis_title='Force (kN)',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    plots['envelope_comparison'] = json.loads(fig9.to_json())

    # 10. Ductility Calculation Explanation (Test Data)
    test_ductility_calc = calculate_ductility(dataset2['u'], dataset2['RF'])

    fig10 = go.Figure()

    # Plot the hysteresis curve
    fig10.add_trace(go.Scatter(
        x=dataset2['u'],
        y=dataset2['RF'],
        mode='lines',
        name=test_name,
        line=dict(color='lightgray', width=1),
        opacity=0.6
    ))

    # Add envelope curve
    fig10.add_trace(go.Scatter(
        x=envelope_test['displacement'],
        y=envelope_test['force'],
        mode='lines',
        name='Courbe Enveloppe',
        line=dict(color='blue', width=3)
    ))

    if test_ductility_calc['yield_displacement'] > 0 and test_ductility_calc['ultimate_displacement'] > 0:
        # Calculate ductility value
        ductility_value = test_ductility_calc['ultimate_displacement'] / test_ductility_calc['yield_displacement']

        # Mark yield point
        yield_force = 0.75 * max(abs(max(dataset2['RF'])), abs(min(dataset2['RF'])))
        fig10.add_trace(go.Scatter(
            x=[test_ductility_calc['yield_displacement']],
            y=[yield_force],
            mode='markers',
            name='Point de Plastification (75% F_max)',
            marker=dict(color='green', size=12, symbol='circle')
        ))

        # Mark ultimate displacement point
        ultimate_force_idx = dataset2['u'].abs().idxmax()
        ultimate_force = dataset2['RF'].iloc[ultimate_force_idx]
        fig10.add_trace(go.Scatter(
            x=[test_ductility_calc['ultimate_displacement']],
            y=[ultimate_force],
            mode='markers',
            name='Point Ultime',
            marker=dict(color='red', size=12, symbol='diamond')
        ))

        # Add vertical lines for displacement markers with labels at top
        fig10.add_vline(
            x=test_ductility_calc['yield_displacement'],
            line=dict(color='green', width=2, dash='dash'),
            annotation_text=f"Iδy = {test_ductility_calc['yield_displacement']:.1f} mm",
            annotation_position="top"
        )

        fig10.add_vline(
            x=test_ductility_calc['ultimate_displacement'],
            line=dict(color='red', width=2, dash='dash'),
            annotation_text=f"Iδu = {test_ductility_calc['ultimate_displacement']:.1f} mm",
            annotation_position="top"
        )

        # Add a prominent text box with ductility calculation
        # Position it in a visible location
        max_force = max(abs(max(dataset2['RF'])), abs(min(dataset2['RF'])))
        min_force = min(dataset2['RF'])

        annotation_text = (
            f"<b>Calcul de Ductilité:</b><br>"
            f"μ = δu / δy<br>"
            f"μ = {test_ductility_calc['ultimate_displacement']:.1f} / {test_ductility_calc['yield_displacement']:.1f}<br>"
            f"<b>μ = {ductility_value:.2f}</b>"
        )

        fig10.add_annotation(
            x=0.02,  # Position at 2% from left
            y=0.98,  # Position at 98% from bottom (near top)
            xref="paper",
            yref="paper",
            text=annotation_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=2,
            borderpad=10,
            font=dict(size=12, color="black"),
            align="left",
            xanchor="left",
            yanchor="top"
        )

    fig10.update_layout(
        title=f'{test_name} : Explication du Calcul de Ductilité',
        xaxis_title='Déplacement u (mm)',
        yaxis_title='Force de Réaction RF (kN)',
        hovermode='closest',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    plots['ductility_explanation'] = json.loads(fig10.to_json())

    # 11. Bilinear Idealization (Test Data)
    # Extract key points for bilinear idealization calculation
    key_envelope = extract_key_envelope_points(dataset2['u'], dataset2['RF'])
    bilinear_data = create_bilinear_idealization(
        key_envelope['displacement'],
        key_envelope['force']
    )

    if bilinear_data:
        fig11 = go.Figure()

        # Plot original hysteresis curve
        fig11.add_trace(go.Scatter(
            x=dataset2['u'],
            y=dataset2['RF'],
            mode='lines',
            name=test_name,
            line=dict(color='lightgray', width=1),
            opacity=0.5
        ))

        # Plot full envelope curve (backbone)
        fig11.add_trace(go.Scatter(
            x=envelope_test['displacement'],
            y=envelope_test['force'],
            mode='lines+markers',
            name='Enveloppe',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))

        # Plot bilinear idealization
        fig11.add_trace(go.Scatter(
            x=bilinear_data['bilinear_disp'],
            y=bilinear_data['bilinear_force'],
            mode='lines+markers',
            name='Idéalisation Bilinéaire',
            line=dict(color='red', width=3),
            marker=dict(size=8, color='red')
        ))

        # Mark yield point
        fig11.add_trace(go.Scatter(
            x=[bilinear_data['yield_displacement']],
            y=[bilinear_data['yield_force']],
            mode='markers',
            name='Point de Plastification',
            marker=dict(color='green', size=12, symbol='circle')
        ))
        
        # Add vertical line at yield
        fig11.add_vline(
            x=bilinear_data['yield_displacement'],
            line=dict(color='green', width=2, dash='dash'),
            annotation_text=f"Iδy = {bilinear_data['yield_displacement']:.1f} mm",
            annotation_position="top"
        )

        # Add vertical line at ultimate
        fig11.add_vline(
            x=bilinear_data['ultimate_displacement'],
            line=dict(color='red', width=2, dash='dash'),
            annotation_text=f"Iδu = {bilinear_data['ultimate_displacement']:.1f} mm",
            annotation_position="top"
        )
        
        # Calculate and display ductility
        ductility = bilinear_data['ultimate_displacement'] / bilinear_data['yield_displacement']

        # Add ductility calculation annotation - positioned in corner to not block graph
        # Use paper coordinates (0-1 range) for consistent positioning
        annotation_text = (
            f"<b>Calcul de Ductilité:</b><br>"
            f"μ = δu / δy<br>"
            f"μ = {bilinear_data['ultimate_displacement']:.1f} / {bilinear_data['yield_displacement']:.1f}<br>"
            f"<b>μ = {ductility:.2f}</b><br>"
            f"Rigidité Initiale : {bilinear_data['initial_stiffness']:.1f} kN/mm"
        )

        fig11.add_annotation(
            x=0.98,  # Position at 98% from left (right side)
            y=0.02,  # Position at 2% from bottom (bottom side)
            xref="paper",
            yref="paper",
            text=annotation_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="black",
            borderwidth=2,
            borderpad=10,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="right",
            yanchor="bottom"
        )
        
        fig11.update_layout(
            title=f'{test_name} : Idéalisation Bilinéaire Force-Déplacement',
            xaxis_title='Déplacement u (mm)',
            yaxis_title='Force de Réaction RF (kN)',
            hovermode='closest',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        plots['bilinear_idealization'] = json.loads(fig11.to_json())

    return plots


def calculate_behavior_factor(displacement, force, ductility_data):
    """Calculate behavior factor (q-factor) for seismic design"""
    import numpy as np
    
    try:
        disp = np.array(displacement)
        force_vals = np.array(force)
        
        # Get ductility from previous calculation
        mu = ductility_data['displacement_ductility']
        
        if mu <= 0:
            return {'q_factor': 0, 'overstrength_factor': 0, 'ductility_reduction_factor': 0}
        
        # Calculate overstrength factor (Ω)
        # Ratio of maximum force to yield force
        max_force = np.max(np.abs(force_vals))
        yield_force = 0.75 * max_force  # Using 75% definition for yield
        overstrength_factor = max_force / yield_force if yield_force > 0 else 1.0
        
        # Calculate ductility reduction factor (Rμ)
        # Based on different structural systems and seismic codes
        if mu <= 1.5:
            # Brittle behavior
            ductility_reduction_factor = 1.0
        elif mu <= 4.0:
            # Limited ductility - bilinear relationship
            ductility_reduction_factor = 1.0 + (mu - 1.0) * 0.5
        else:
            # High ductility - plateau effect
            ductility_reduction_factor = 1.5 + np.sqrt(mu - 1.0)
        
        # Calculate total behavior factor
        # q = Ω × Rμ (simplified approach)
        q_factor = overstrength_factor * ductility_reduction_factor
        
        # Apply practical limits (typical range 1.0 to 8.0)
        q_factor = max(1.0, min(8.0, q_factor))
        
        return {
            'q_factor': float(q_factor),
            'overstrength_factor': float(overstrength_factor),
            'ductility_reduction_factor': float(ductility_reduction_factor),
            'ductility_class': classify_ductility_class(mu, q_factor)
        }
    except:
        return {'q_factor': 0, 'overstrength_factor': 0, 'ductility_reduction_factor': 0, 'ductility_class': 'Unknown'}


def classify_ductility_class(mu, q):
    """Classify ductility based on μ and q-factor according to Eurocode 8"""
    if q >= 4.0 and mu >= 4.0:
        return 'DCH (High Ductility)'
    elif q >= 2.5 and mu >= 2.5:
        return 'DCM (Medium Ductility)'
    elif q >= 1.5 and mu >= 1.5:
        return 'DCL (Limited Ductility)'
    else:
        return 'Brittle Behavior'


def create_bilinear_idealization(displacement, force):
    """Create bilinear idealization from unified envelope curve (backbone) with perfectly straight line segments"""
    import numpy as np

    try:
        # Convert to numpy arrays - these are already from the unified envelope curve
        disp = np.array(displacement)
        force_vals = np.array(force)

        # Find maximum displacement and force points from envelope
        max_disp_idx = np.argmax(np.abs(disp))
        ultimate_disp = np.abs(disp[max_disp_idx])
        ultimate_force = force_vals[max_disp_idx]

        # Find initial stiffness from the first few points of envelope (elastic region)
        # Sort by displacement first
        sorted_indices = np.argsort(np.abs(disp))
        sorted_disp = disp[sorted_indices]
        sorted_force = force_vals[sorted_indices]

        # Use first 15% of envelope points for initial stiffness calculation
        n_initial = max(5, len(sorted_disp) // 7)
        initial_points_disp = sorted_disp[:n_initial]
        initial_points_force = sorted_force[:n_initial]

        # Calculate initial stiffness (elastic slope) from envelope
        if len(initial_points_disp) > 1:
            # Remove points too close to zero
            mask = np.abs(initial_points_disp) > 0.01 * ultimate_disp
            if np.any(mask):
                initial_points_disp = initial_points_disp[mask]
                initial_points_force = initial_points_force[mask]

            if len(initial_points_disp) > 1:
                initial_slope = np.polyfit(initial_points_disp, initial_points_force, 1)[0]
            else:
                initial_slope = ultimate_force / ultimate_disp * 0.8
        else:
            initial_slope = ultimate_force / ultimate_disp * 0.8

        # Define yield displacement using energy equivalence method
        # Area under bilinear curve should equal area under envelope curve

        # Calculate area under envelope curve
        total_energy = np.trapz(np.abs(force_vals), np.abs(disp))

        # For bilinear: Area = 0.5 * yield_force * yield_disp + (ultimate_disp - yield_disp) * ultimate_force
        # Where yield_force = initial_slope * yield_disp
        # Solving for yield_disp using energy equivalence

        # Simplified approach: Use typical yield displacement (around 60-80% of elastic limit)
        elastic_limit_disp = ultimate_force / initial_slope if initial_slope > 0 else ultimate_disp * 0.5
        yield_disp = min(elastic_limit_disp * 0.75, ultimate_disp * 0.6)
        yield_force = initial_slope * yield_disp

        # Ensure yield point is reasonable
        if yield_disp <= 0 or yield_disp >= ultimate_disp:
            yield_disp = ultimate_disp * 0.4
            yield_force = initial_slope * yield_disp

        # Create perfectly straight bilinear idealization with just 3 points
        # This creates two straight line segments from the unified envelope (backbone)
        bilinear_disp = np.array([0, yield_disp, ultimate_disp])
        bilinear_force = np.array([0, yield_force, ultimate_force])

        return {
            'envelope_disp': disp,
            'envelope_force': force_vals,
            'bilinear_disp': bilinear_disp,
            'bilinear_force': bilinear_force,
            'yield_displacement': yield_disp,
            'yield_force': yield_force,
            'ultimate_displacement': ultimate_disp,
            'ultimate_force': ultimate_force,
            'initial_stiffness': initial_slope
        }

    except Exception as e:
        return None


def calculate_ductility(displacement, force):
    """Calculate ductility from force-displacement data"""
    import numpy as np
    
    try:
        disp = np.array(displacement)
        force_vals = np.array(force)
        
        # Find yield point (approximated as 75% of maximum force)
        max_force = np.max(np.abs(force_vals))
        yield_force = 0.75 * max_force
        
        # Find yield displacement (first occurrence of yield force)
        yield_indices = np.where(np.abs(force_vals) >= yield_force)[0]
        if len(yield_indices) == 0:
            return {'displacement_ductility': 0, 'yield_displacement': 0, 'ultimate_displacement': 0}
        
        yield_displacement = np.abs(disp[yield_indices[0]])
        
        # Find ultimate displacement (maximum displacement)
        ultimate_displacement = np.max(np.abs(disp))
        
        # Calculate displacement ductility
        displacement_ductility = ultimate_displacement / yield_displacement if yield_displacement > 0 else 0
        
        return {
            'displacement_ductility': float(displacement_ductility),
            'yield_displacement': float(yield_displacement),
            'ultimate_displacement': float(ultimate_displacement)
        }
    except:
        return {'displacement_ductility': 0, 'yield_displacement': 0, 'ultimate_displacement': 0}


def calculate_engineering_metrics(data):
    """Calculate engineering metrics from the datasets"""
    dataset1 = data['dataset1']
    dataset2 = data['dataset2']

    metrics = {}

    # Reference case metrics
    ref_displacement_range = dataset1['U_mm'].max() - dataset1['U_mm'].min()
    ref_force_range = dataset1['F_kn'].max() - dataset1['F_kn'].min()

    # Test case metrics
    test_displacement_range = dataset2['u'].max() - dataset2['u'].min()
    test_force_range = dataset2['RF'].max() - dataset2['RF'].min()

    # Calculate approximate stiffness (initial slope)
    ref_stiffness = abs(dataset1['F_kn'].iloc[5] - dataset1['F_kn'].iloc[1]) / abs(dataset1['U_mm'].iloc[5] - dataset1['U_mm'].iloc[1]) if len(dataset1) > 5 else 0
    test_stiffness = abs(dataset2['RF'].iloc[5] - dataset2['RF'].iloc[1]) / abs(dataset2['u'].iloc[5] - dataset2['u'].iloc[1]) if len(dataset2) > 5 else 0

    # Energy dissipation (total)
    ref_energy = sum([abs(dataset1['F_kn'].iloc[i] * (dataset1['U_mm'].iloc[i] - dataset1['U_mm'].iloc[i-1])) for i in range(1, len(dataset1))])
    test_energy = sum([abs(dataset2['RF'].iloc[i] * (dataset2['u'].iloc[i] - dataset2['u'].iloc[i-1])) for i in range(1, len(dataset2))])
    
    # Calculate ductility
    ref_ductility = calculate_ductility(dataset1['U_mm'], dataset1['F_kn'])
    test_ductility = calculate_ductility(dataset2['u'], dataset2['RF'])
    
    # Calculate behavior factors
    ref_behavior = calculate_behavior_factor(dataset1['U_mm'], dataset1['F_kn'], ref_ductility)
    test_behavior = calculate_behavior_factor(dataset2['u'], dataset2['RF'], test_ductility)

    metrics['reference'] = {
        'displacement_range': float(ref_displacement_range),
        'force_range': float(ref_force_range),
        'stiffness': float(ref_stiffness),
        'total_energy': float(ref_energy),
        'max_force': float(dataset1['F_kn'].max()),
        'min_force': float(dataset1['F_kn'].min()),
        'max_displacement': float(dataset1['U_mm'].max()),
        'min_displacement': float(dataset1['U_mm'].min()),
        'displacement_ductility': ref_ductility['displacement_ductility'],
        'yield_displacement': ref_ductility['yield_displacement'],
        'ultimate_displacement': ref_ductility['ultimate_displacement'],
        'q_factor': ref_behavior['q_factor'],
        'overstrength_factor': ref_behavior['overstrength_factor'],
        'ductility_reduction_factor': ref_behavior['ductility_reduction_factor'],
        'ductility_class': ref_behavior['ductility_class']
    }

    metrics['test'] = {
        'displacement_range': float(test_displacement_range),
        'force_range': float(test_force_range),
        'stiffness': float(test_stiffness),
        'total_energy': float(test_energy),
        'max_force': float(dataset2['RF'].max()),
        'min_force': float(dataset2['RF'].min()),
        'max_displacement': float(dataset2['u'].max()),
        'min_displacement': float(dataset2['u'].min()),
        'displacement_ductility': test_ductility['displacement_ductility'],
        'yield_displacement': test_ductility['yield_displacement'],
        'ultimate_displacement': test_ductility['ultimate_displacement'],
        'q_factor': test_behavior['q_factor'],
        'overstrength_factor': test_behavior['overstrength_factor'],
        'ductility_reduction_factor': test_behavior['ductility_reduction_factor'],
        'ductility_class': test_behavior['ductility_class']
    }

    # Comparative metrics
    metrics['comparison'] = {
        'stiffness_ratio': float((test_stiffness / ref_stiffness * 100) if ref_stiffness != 0 else 0),
        'energy_ratio': float((test_energy / ref_energy * 100) if ref_energy != 0 else 0),
        'displacement_ratio': float((test_displacement_range / ref_displacement_range * 100) if ref_displacement_range != 0 else 0),
        'force_ratio': float((test_force_range / ref_force_range * 100) if ref_force_range != 0 else 0),
        'ductility_ratio': float((test_ductility['displacement_ductility'] / ref_ductility['displacement_ductility'] * 100) if ref_ductility['displacement_ductility'] != 0 else 0),
        'q_factor_ratio': float((test_behavior['q_factor'] / ref_behavior['q_factor'] * 100) if ref_behavior['q_factor'] != 0 else 0)
    }

    return metrics


async def generate_ai_report(test_name, stats, metrics):
    """Generate AI-powered analysis report using OpenAI"""
    try:

        # Prepare data summary for AI
        prompt = f"""You are a structural engineering expert specializing in ductility analysis of cyclic load test results.

Provide a comprehensive ductility analysis report for the following test specimen:

TEST SPECIMEN: {test_name}

DUCTILITY ANALYSIS DATA:

REFERENCE CASE:
- Displacement ductility: {metrics['reference']['displacement_ductility']:.2f}
- Yield displacement: {metrics['reference']['yield_displacement']:.2f} mm
- Ultimate displacement: {metrics['reference']['ultimate_displacement']:.2f} mm

TEST SPECIMEN ({test_name}):
- Displacement ductility: {metrics['test']['displacement_ductility']:.2f}
- Yield displacement: {metrics['test']['yield_displacement']:.2f} mm
- Ultimate displacement: {metrics['test']['ultimate_displacement']:.2f} mm

DUCTILITY COMPARISON:
- Ductility ratio (test/reference): {metrics['comparison']['ductility_ratio']:.1f}%

Please provide a focused ductility analysis report with the following sections:

1. **Ductility Assessment**: Evaluate the displacement ductility values and what they indicate about structural behavior
2. **Yield Point Analysis**: Discuss the yield displacement characteristics and implications
3. **Ultimate Capacity**: Analyze the ultimate displacement and deformation capacity
4. **Comparative Ductility**: Compare test specimen ductility vs reference case
5. **Ductility Classification**: Classify the structural behavior (highly ductile, moderately ductile, limited ductility, brittle)
6. **Engineering Implications**: What do these ductility values mean for structural performance and seismic design?
7. **Recommendations**: Specific recommendations based on ductility performance

Focus exclusively on ductility-related aspects and structural deformation characteristics. Format the response in clear sections with markdown formatting."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert structural engineer specializing in experimental testing and seismic performance evaluation. Provide detailed, technically accurate analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI Report generation failed: {str(e)}\n\nPlease check your OpenAI API key configuration."


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload Excel file and return processed data (no plots)"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Please upload an Excel file (.xlsx or .xls)")

    try:
        content = await file.read()
        data = process_excel_file(content)

        # Calculate statistics
        stats = {
            'reference': {
                'points': len(data['dataset1']),
                'max_displacement': float(data['dataset1']['U_mm'].max()),
                'min_displacement': float(data['dataset1']['U_mm'].min()),
                'max_force': float(data['dataset1']['F_kn'].max()),
                'min_force': float(data['dataset1']['F_kn'].min()),
            },
            'test': {
                'name': data['test_name'],
                'points': len(data['dataset2']),
                'max_displacement': float(data['dataset2']['u'].max()),
                'min_displacement': float(data['dataset2']['u'].min()),
                'max_force': float(data['dataset2']['RF'].max()),
                'min_force': float(data['dataset2']['RF'].min()),
            }
        }

        # Calculate engineering metrics
        metrics = calculate_engineering_metrics(data)

        # Convert datasets to JSON-serializable format
        dataset1_json = data['dataset1'].to_dict('list')
        dataset2_json = data['dataset2'].to_dict('list')

        return JSONResponse({
            'success': True,
            'test_name': data['test_name'],
            'dataset1': dataset1_json,
            'dataset2': dataset2_json,
            'stats': stats,
            'metrics': metrics
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def answer_simple_query(query: str, session_data: dict):
    """Answer simple data queries without AI"""
    query_lower = query.lower()

    if not session_data or 'stats' not in session_data:
        return None

    stats = session_data.get('stats', {})
    metrics = session_data.get('metrics', {})

    # Max/Min force queries
    if 'max force' in query_lower or 'maximum force' in query_lower:
        if 'test' in query_lower:
            return f"The maximum force in the test data is **{stats['test']['max_force']:.2f} kN**."
        elif 'reference' in query_lower:
            return f"The maximum force in the reference case is **{stats['reference']['max_force']:.2f} kN**."
        else:
            return f"Maximum forces:\n- Reference: **{stats['reference']['max_force']:.2f} kN**\n- Test: **{stats['test']['max_force']:.2f} kN**"

    if 'min force' in query_lower or 'minimum force' in query_lower:
        if 'test' in query_lower:
            return f"The minimum force in the test data is **{stats['test']['min_force']:.2f} kN**."
        elif 'reference' in query_lower:
            return f"The minimum force in the reference case is **{stats['reference']['min_force']:.2f} kN**."
        else:
            return f"Minimum forces:\n- Reference: **{stats['reference']['min_force']:.2f} kN**\n- Test: **{stats['test']['min_force']:.2f} kN**"

    # Displacement queries
    if 'max displacement' in query_lower or 'maximum displacement' in query_lower:
        if 'test' in query_lower:
            return f"The maximum displacement in the test data is **{stats['test']['max_displacement']:.2f} mm**."
        elif 'reference' in query_lower:
            return f"The maximum displacement in the reference case is **{stats['reference']['max_displacement']:.2f} mm**."
        else:
            return f"Maximum displacements:\n- Reference: **{stats['reference']['max_displacement']:.2f} mm**\n- Test: **{stats['test']['max_displacement']:.2f} mm**"

    # Ductility queries
    if 'ductility' in query_lower:
        if 'test' in query_lower:
            return f"Test displacement ductility: **{metrics['test']['displacement_ductility']:.2f}**\nYield displacement: **{metrics['test']['yield_displacement']:.2f} mm**\nUltimate displacement: **{metrics['test']['ultimate_displacement']:.2f} mm**"
        elif 'reference' in query_lower:
            return f"Reference displacement ductility: **{metrics['reference']['displacement_ductility']:.2f}**\nYield displacement: **{metrics['reference']['yield_displacement']:.2f} mm**\nUltimate displacement: **{metrics['reference']['ultimate_displacement']:.2f} mm**"
        else:
            return f"Displacement ductility comparison:\n- Reference: **{metrics['reference']['displacement_ductility']:.2f}**\n- Test: **{metrics['test']['displacement_ductility']:.2f}**\n- Ratio: **{metrics['comparison']['ductility_ratio']:.1f}%**"

    # Stiffness queries
    if 'stiffness' in query_lower:
        return f"Stiffness comparison:\n- Reference: **{metrics['reference']['stiffness']:.2f} kN/mm**\n- Test: **{metrics['test']['stiffness']:.2f} kN/mm**\n- Ratio: **{metrics['comparison']['stiffness_ratio']:.1f}%**"

    # Energy queries
    if 'energy' in query_lower:
        return f"Total energy dissipation:\n- Reference: **{metrics['reference']['total_energy']:.2f} kN·mm**\n- Test: **{metrics['test']['total_energy']:.2f} kN·mm**\n- Ratio: **{metrics['comparison']['energy_ratio']:.1f}%**"

    return None


def generate_plot_for_query(query: str, session_data: dict):
    """Generate specific plot based on query"""
    if not session_data or 'dataset1' not in session_data:
        return None

    query_lower = query.lower()

    # Reconstruct data from session
    import pandas as pd
    dataset1 = pd.DataFrame(session_data['dataset1'])
    dataset2 = pd.DataFrame(session_data['dataset2'])
    test_name = session_data.get('test_name', 'Test Data')

    data = {
        'dataset1': dataset1,
        'dataset2': dataset2,
        'test_name': test_name
    }

    # Check if user is asking for plots/graphs (English and French keywords)
    asking_for_visual = any(word in query_lower for word in [
        'show', 'display', 'plot', 'graph', 'chart', 'visualize', 'see',
        'afficher', 'affiche', 'tracer', 'graphique', 'courbe', 'visualiser', 'voir', 'dessiner', 'dessine'
    ])

    # Determine which plot to generate
    plot_type = None

    # Check for "all" plots request or general plot requests
    if asking_for_visual and ('all' in query_lower or 
                              ('graphs' in query_lower and not any(specific in query_lower for specific in ['hysteresis', 'energy', 'force', 'envelope', 'ductility', 'bilinear', 'loading', 'comparison'])) or
                              ('plots' in query_lower and not any(specific in query_lower for specific in ['hysteresis', 'energy', 'force', 'envelope', 'ductility', 'bilinear', 'loading', 'comparison'])) or
                              ('charts' in query_lower and not any(specific in query_lower for specific in ['hysteresis', 'energy', 'force', 'envelope', 'ductility', 'bilinear', 'loading', 'comparison']))):
        all_plots = create_plots(data)
        return {
            'type': 'multiple',
            'plots': all_plots,
            'message': 'Here are all the available plots for your data:'
        }

    # Envelope curves (Backbone curves) - Check this FIRST before other curve types
    if asking_for_visual and ('envelope' in query_lower or 'backbone' in query_lower or ('unified' in query_lower and 'envelope' in query_lower)):
        if 'comparison' in query_lower or 'compare' in query_lower or 'both' in query_lower:
            plot_type = 'envelope_comparison'
        elif 'test' in query_lower or any(test_word in query_lower for test_word in ['bcjs', 'specimen']):
            plot_type = 'envelope_test'
        elif 'reference' in query_lower:
            plot_type = 'envelope_ref'
        else:
            plot_type = 'envelope_comparison'

    # Hysteresis curves - be more specific about which one to show
    elif asking_for_visual and ('hysteresis' in query_lower or 'force-displacement' in query_lower or 'force displacement' in query_lower):
        if 'test' in query_lower or any(test_word in query_lower for test_word in ['bcjs', 'specimen']):
            plot_type = 'hysteresis_test'
        elif 'reference' in query_lower:
            plot_type = 'hysteresis_ref'
        elif 'comparison' in query_lower or 'compare' in query_lower or 'both' in query_lower or 'vs' in query_lower:
            plot_type = 'comparison'
        else:
            # If just "force displacement curve" without specifying which, show the reference case first
            plot_type = 'hysteresis_ref'
    
    # Force-displacement curve specific requests (but not envelope curves)
    # Support both English and French keywords
    elif asking_for_visual and \
         (('force' in query_lower and ('displacement' in query_lower or 'déplacement' in query_lower or 'deplacement' in query_lower)) or \
          ('curve' in query_lower or 'courbe' in query_lower) and ('force' in query_lower or 'displacement' in query_lower or 'déplacement' in query_lower or 'deplacement' in query_lower)) and \
         not ('envelope' in query_lower or 'backbone' in query_lower or 'enveloppe' in query_lower):
        if 'test' in query_lower or any(test_word in query_lower for test_word in ['bcjs', 'specimen', 'essai']):
            plot_type = 'hysteresis_test'
        elif 'reference' in query_lower or 'référence' in query_lower:
            plot_type = 'hysteresis_ref'
        elif 'comparison' in query_lower or 'compare' in query_lower or 'vs' in query_lower or 'comparaison' in query_lower:
            plot_type = 'comparison'
        else:
            # Default to reference case for generic "force displacement curve"
            plot_type = 'hysteresis_ref'

    # Comparison plot
    elif asking_for_visual and ('comparison' in query_lower or 'compare' in query_lower):
        plot_type = 'comparison'

    # Loading/Displacement history
    elif asking_for_visual and ('loading' in query_lower and 'history' in query_lower):
        plot_type = 'loading_history'
    elif asking_for_visual and ('displacement' in query_lower and ('history' in query_lower or 'time' in query_lower)):
        plot_type = 'loading_history'

    # Force history
    elif asking_for_visual and ('force' in query_lower and 'history' in query_lower):
        plot_type = 'force_history'

    # Energy dissipation / Cumulative Energy
    elif asking_for_visual and (('energy' in query_lower and ('dissipation' in query_lower or 'cumulative' in query_lower)) or 'cumulative energy' in query_lower):
        plot_type = 'energy_dissipation'


    # Ductility explanation - expanded to include more keywords
    elif asking_for_visual and 'ductility' in query_lower:
        # Check if asking specifically for bilinear (handled below), otherwise show ductility explanation
        if not ('bilinear' in query_lower or 'idealization' in query_lower):
            plot_type = 'ductility_explanation'

    # Bilinear idealization
    elif asking_for_visual and ('bilinear' in query_lower or 'idealization' in query_lower):
        plot_type = 'bilinear_idealization'

    if plot_type:
        plots = create_plots(data)
        if plot_type in plots:
            return {
                'type': 'single',
                'plot': plots[plot_type],
                'plot_name': plot_type
            }

    return None


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle chat messages with intelligent routing"""
    try:
        message = request.message
        session_data = request.session_data

        # Check if file is uploaded
        if not session_data or 'stats' not in session_data:
            return JSONResponse({
                'success': True,
                'response': 'Please upload an Excel file first to start analyzing your data.',
                'type': 'text'
            })

        # Check if query is asking for a plot FIRST (prioritize visual requests)
        plot_result = generate_plot_for_query(message, session_data)
        if plot_result:
            if plot_result['type'] == 'single':
                return JSONResponse({
                    'success': True,
                    'response': f"Here's the {plot_result['plot_name'].replace('_', ' ')} plot:",
                    'type': 'plot',
                    'plot': plot_result['plot']
                })
            elif plot_result['type'] == 'multiple':
                return JSONResponse({
                    'success': True,
                    'response': plot_result['message'],
                    'type': 'plots',
                    'plots': plot_result['plots']
                })

        # Try simple data queries only if no plot was requested
        simple_answer = answer_simple_query(message, session_data)
        if simple_answer:
            return JSONResponse({
                'success': True,
                'response': simple_answer,
                'type': 'text'
            })

        # Use AI for complex queries and interpretations
        try:
            stats = session_data.get('stats', {})
            metrics = session_data.get('metrics', {})
            test_name = session_data.get('test_name', 'Test Data')

            system_prompt = """You are an expert structural engineer specializing in cyclic loading tests, ductility analysis, and seismic performance evaluation.

You have access to test data including:
- Force-displacement relationships
- Ductility metrics
- Energy dissipation
- Stiffness values
- Behavior factors

Provide clear, concise, and technically accurate answers. Use markdown formatting for emphasis (**bold**, *italic*). Keep responses focused and practical."""

            user_prompt = f"""Based on the following structural test data, answer this question: "{message}"

TEST DATA SUMMARY:
Test Name: {test_name}

REFERENCE CASE:
- Max Force: {stats['reference']['max_force']:.2f} kN
- Min Force: {stats['reference']['min_force']:.2f} kN
- Max Displacement: {stats['reference']['max_displacement']:.2f} mm
- Min Displacement: {stats['reference']['min_displacement']:.2f} mm
- Displacement Ductility: {metrics['reference']['displacement_ductility']:.2f}
- Yield Displacement: {metrics['reference']['yield_displacement']:.2f} mm
- Ultimate Displacement: {metrics['reference']['ultimate_displacement']:.2f} mm
- Stiffness: {metrics['reference']['stiffness']:.2f} kN/mm
- Total Energy: {metrics['reference']['total_energy']:.2f} kN·mm
- Q-factor: {metrics['reference']['q_factor']:.2f}
- Ductility Class: {metrics['reference']['ductility_class']}

TEST DATA:
- Max Force: {stats['test']['max_force']:.2f} kN
- Min Force: {stats['test']['min_force']:.2f} kN
- Max Displacement: {stats['test']['max_displacement']:.2f} mm
- Min Displacement: {stats['test']['min_displacement']:.2f} mm
- Displacement Ductility: {metrics['test']['displacement_ductility']:.2f}
- Yield Displacement: {metrics['test']['yield_displacement']:.2f} mm
- Ultimate Displacement: {metrics['test']['ultimate_displacement']:.2f} mm
- Stiffness: {metrics['test']['stiffness']:.2f} kN/mm
- Total Energy: {metrics['test']['total_energy']:.2f} kN·mm
- Q-factor: {metrics['test']['q_factor']:.2f}
- Ductility Class: {metrics['test']['ductility_class']}

COMPARISON:
- Stiffness Ratio: {metrics['comparison']['stiffness_ratio']:.1f}%
- Energy Ratio: {metrics['comparison']['energy_ratio']:.1f}%
- Ductility Ratio: {metrics['comparison']['ductility_ratio']:.1f}%

Please provide a clear and concise answer."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            ai_response = response.choices[0].message.content

            return JSONResponse({
                'success': True,
                'response': ai_response,
                'type': 'text'
            })

        except Exception as e:
            return JSONResponse({
                'success': True,
                'response': f"I encountered an error while processing your question: {str(e)}",
                'type': 'text'
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
