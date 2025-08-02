// Log to confirm script execution
console.log('interactive.js loaded at', new Date().toISOString());

// Ensure DOM and Streamlit data are ready
function whenReady(callback) {
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        if (window.streamlitDataReady) {
            console.log('DOM and Streamlit data ready');
            callback();
        } else {
            console.log('Waiting for Streamlit data...');
            setTimeout(() => whenReady(callback), 100);
        }
    } else {
        console.log('Waiting for DOM...');
        document.addEventListener('DOMContentLoaded', () => whenReady(callback));
    }
}

// Function to create clickable hypothesis ranking list
function initHypothesisList(containerId, hypotheses, onClickCallback) {
    try {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            document.getElementById('hypothesis-list-fallback').innerText = 'Error: Hypothesis container not found';
            return;
        }

        // Validate hypotheses
        if (!Array.isArray(hypotheses) || hypotheses.length === 0) {
            console.warn('No valid hypotheses provided');
            document.getElementById('hypothesis-list-fallback').innerText = 'No hypotheses available';
            return;
        }

        // Clear existing content
        container.innerHTML = '';

        // Create list
        const ul = document.createElement('ul');
        ul.style.listStyle = 'none';
        ul.style.padding = '0';
        ul.style.margin = '0';

        hypotheses.forEach(hyp => {
            if (!hyp.hypothesis || typeof hyp.score !== 'number') {
                console.warn('Invalid hypothesis:', hyp);
                return;
            }
            const li = document.createElement('li');
            li.style.padding = '10px';
            li.style.marginBottom = '5px';
            li.style.backgroundColor = '#ffffff';
            li.style.borderRadius = '4px';
            li.style.cursor = 'pointer';
            li.style.transition = 'background-color 0.2s';
            li.textContent = `${hyp.hypothesis} (Score: ${hyp.score.toFixed(3)})`;
            li.addEventListener('mouseover', () => {
                li.style.backgroundColor = '#e8f0fe';
            });
            li.addEventListener('mouseout', () => {
                li.style.backgroundColor = '#ffffff';
            });
            li.addEventListener('click', () => onClickCallback(hyp));
            ul.appendChild(li);
        });

        container.appendChild(ul);
        console.log(`Initialized hypothesis list with ${hypotheses.length} items`);
    } catch (error) {
        console.error('Error initializing hypothesis list:', error);
        document.getElementById('hypothesis-list-fallback').innerText = 'Error loading hypotheses';
    }
}

// Initialize dashboard interactivity
whenReady(() => {
    window.streamlitDataReady = true;
    console.log('Initializing interactivity with data:', {
        proteinData: window.proteinData,
        hypotheses: window.hypotheses
    });

    // Skip protein visualization (removed Three.js for simplicity)
    const proteinFallback = document.getElementById('protein-viz-fallback');
    if (proteinFallback) {
        proteinFallback.innerText = '3D protein visualization disabled for debugging';
    }

    // Initialize hypothesis list
    const hypotheses = window.hypotheses || [];
    if (document.getElementById('hypothesis-list') && hypotheses.length > 0) {
        initHypothesisList('hypothesis-list', hypotheses, (hypothesis) => {
            alert(`Selected: ${hypothesis.hypothesis}\nScore: ${hypothesis.score.toFixed(3)}`);
            console.log('Hypothesis clicked:', hypothesis);
        });
    } else {
        console.warn('Hypothesis list not initialized: missing container or data');
        if (document.getElementById('hypothesis-list-fallback')) {
            document.getElementById('hypothesis-list-fallback').innerText = 'No hypotheses available';
        }
    }
});

// Handle Streamlit render events
document.addEventListener('streamlit:render', (event) => {
    try {
        console.log('Streamlit render event received:', event.detail);
        const data = event.detail.args || {};
        window.proteinData = data.proteinData || [];
        window.hypotheses = data.hypotheses || [];
        window.streamlitDataReady = true;

        // Reinitialize hypothesis list
        if (document.getElementById('hypothesis-list') && window.hypotheses.length > 0) {
            initHypothesisList('hypothesis-list', window.hypotheses, (hypothesis) => {
                alert(`Selected: ${hypothesis.hypothesis}\nScore: ${hypothesis.score.toFixed(3)}`);
            });
        }

        // Update view
        const view = data.view || 'Simulations';
        document.getElementById('simulations-view').style.display = view === 'Simulations' ? 'block' : 'none';
        document.getElementById('hypotheses-view').style.display = view === 'Hypotheses' ? 'block' : 'none';
    } catch (error) {
        console.error('Error handling Streamlit render event:', error);
        if (document.getElementById('hypothesis-list-fallback')) {
            document.getElementById('hypothesis-list-fallback').innerText = 'Error loading hypotheses';
        }
    }
});
