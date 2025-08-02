import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';
import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.146.0/examples/jsm/controls/OrbitControls.js';
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.146.0/build/three.module.js';

// Function to initialize 3D protein visualization
function initProteinVisualization(containerId, data) {
    try {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        // Set up scene
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        // Add orbit controls for zooming and rotation
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        controls.screenSpacePanning = false;
        controls.maxDistance = 100;

        // Parse data (expecting array of [x, y, z] coordinates)
        const coordinates = data.map(d => [parseFloat(d[0]), parseFloat(d[1]), parseFloat(d[2])]).filter(d => !isNaN(d[0]));

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
            .text(d => `${d.hypothesis} (Score: ${d.score.toFixed(3)})`)
            .on('mouseover', function() {
                d3.select(this).style('background-color', '#e8f0fe');
            })
            .on('mouseout', function() {
                d3.select(this).style('background-color', '#ffffff');
            })
            .on('click', function(event, d) {
                onClickCallback(d);
            });
    } catch (error) {
        console.error('Error initializing hypothesis list:', error);
    }
}

// Initialize dashboard interactivity
document.addEventListener('DOMContentLoaded', () => {
    // Example: Fetch simulation data (mocked here, replace with actual fetch from Streamlit)
    const proteinData = window.proteinData || []; // Expected format: [[x, y, z], ...]
    const hypotheses = window.hypotheses || []; // Expected format: [{hypothesis, score, ...}, ...]

    // Initialize protein visualization
    if (proteinData.length > 0) {
        initProteinVisualization('protein-viz', proteinData);
    }

    // Initialize hypothesis list
    if (hypotheses.length > 0) {
        initHypothesisList('hypothesis-list', hypotheses, (hypothesis) => {
            // Callback for hypothesis click
            alert(`Selected: ${hypothesis.hypothesis}\nScore: ${hypothesis.score.toFixed(3)}`);
            // Add logic to update dashboard (e.g., highlight related simulation)
        });
    }
});

// Export functions for external use
export { initProteinVisualization, initHypothesisList };
