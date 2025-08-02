import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';
import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.146.0/examples/jsm/controls/OrbitControls.js';
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.146.0/build/three.module.js';

// Ensure DOM is ready before executing
function whenDocumentReady(callback) {
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        callback();
    } else {
        document.addEventListener('DOMContentLoaded', callback);
    }
}

// Function to initialize 3D protein visualization
function initProteinVisualization(containerId, data) {
    try {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        // Validate data
        if (!Array.isArray(data) || data.length === 0) {
            console.warn('No valid protein data provided');
            return;
        }

        // Set up scene
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.innerHTML = ''; // Clear container
        container.appendChild(renderer.domElement);

        // Add orbit controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        controls.screenSpacePanning = false;
        controls.maxDistance = 100;

        // Parse and validate coordinates
        const coordinates = data
            .map(d => {
                if (Array.isArray(d) && d.length >= 3) {
                    return [parseFloat(d[0]), parseFloat(d[1]), parseFloat(d[2])];
                }
                return null;
            })
            .filter(d => d && d.every(v => !isNaN(v)));

        if (coordinates.length === 0) {
            console.warn('No valid coordinates for protein visualization');
            return;
        }

        // Create geometry for protein structure
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(coordinates.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

        // Create material and line
        const material = new THREE.LineBasicMaterial({ color: 0x1a73e8 });
        const line = new THREE.Line(geometry, material);
        scene.add(line);

        // Create spheres for C-alpha atoms
        const sphereGeometry = new THREE.SphereGeometry(0.5, 16, 16);
        const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0x4682b4 });
        coordinates.forEach(coord => {
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
            sphere.position.set(coord[0], coord[1], coord[2]);
            scene.add(sphere);
        });

        // Position camera
        camera.position.z = 50;

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
    } catch (error) {
        console.error('Error initializing protein visualization:', error);
    }
}

// Function to create clickable hypothesis ranking list
function initHypothesisList(containerId, hypotheses, onClickCallback) {
    try {
        const container = d3.select(`#${containerId}`);
        if (container.empty()) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        // Validate hypotheses
        if (!Array.isArray(hypotheses) || hypotheses.length === 0) {
            console.warn('No valid hypotheses provided');
            return;
        }

        // Clear existing content
        container.selectAll('*').remove();

        // Create list
        const list = container.append('ul')
            .style('list-style', 'none')
            .style('padding', '0')
            .style('margin', '0');

        list.selectAll('li')
            .data(hypotheses)
            .enter()
            .append('li')
            .style('padding', '10px')
            .style('margin-bottom', '5px')
            .style('background-color', '#ffffff')
            .style('border-radius', '4px')
            .style('cursor', 'pointer')
            .style('transition', 'background-color 0.2s')
            .text(d => d.hypothesis && d.score ? `${d.hypothesis} (Score: ${d.score.toFixed(3)})` : 'Invalid hypothesis')
            .on('mouseover', function() {
                d3.select(this).style('background-color', '#e8f0fe');
            })
            .on('mouseout', function() {
                d3.select(this).style('background-color', '#ffffff');
            })
            .on('click', function(event, d) {
                if (d && d.hypothesis) {
                    onClickCallback(d);
                }
            });
    } catch (error) {
        console.error('Error initializing hypothesis list:', error);
    }
}

// Initialize dashboard interactivity when DOM is ready
whenDocumentReady(() => {
    // Fetch data (Streamlit will inject these variables)
    const proteinData = window.proteinData || [];
    const hypotheses = window.hypotheses || [];

    // Initialize protein visualization
    if (document.getElementById('protein-viz') && proteinData.length > 0) {
        initProteinVisualization('protein-viz', proteinData);
    } else {
        console.warn('Protein visualization not initialized: missing container or data');
    }

    // Initialize hypothesis list
    if (document.getElementById('hypothesis-list') && hypotheses.length > 0) {
        initHypothesisList('hypothesis-list', hypotheses, (hypothesis) => {
            // Callback for hypothesis click
            alert(`Selected: ${hypothesis.hypothesis}\nScore: ${hypothesis.score.toFixed(3)}`);
            // Add logic to update dashboard if needed
        });
    } else {
        console.warn('Hypothesis list not initialized: missing container or data');
    }
});

// Handle Streamlit render events
document.addEventListener('streamlit:render', (event) => {
    try {
        const data = event.detail.args || {};
        window.proteinData = data.proteinData || [];
        window.hypotheses = data.hypotheses || [];
        
        // Reinitialize visualizations with new data
        if (document.getElementById('protein-viz') && window.proteinData.length > 0) {
            initProteinVisualization('protein-viz', window.proteinData);
        }
        if (document.getElementById('hypothesis-list') && window.hypotheses.length > 0) {
            initHypothesisList('hypothesis-list', window.hypotheses, (hypothesis) => {
                alert(`Selected: ${hypothesis.hypothesis}\nScore: ${hypothesis.score.toFixed(3)}`);
            });
        }

        // Toggle view based on Streamlit selection
        const view = data.view || 'Simulations';
        document.getElementById('simulations-view').style.display = view === 'Simulations' ? 'block' : 'none';
        document.getElementById('hypotheses-view').style.display = view === 'Hypotheses' ? 'block' : 'none';
    } catch (error) {
        console.error('Error handling Streamlit render event:', error);
    }
});
