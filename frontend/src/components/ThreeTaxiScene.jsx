import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FaCube, FaCompass, FaCamera } from 'react-icons/fa6';

const LANDMARKS = [
  { r: 0, c: 0, name: 'R', color: 0xef4444, label: 'Red' },
  { r: 0, c: 4, name: 'G', color: 0x16a34a, label: 'Green' },
  { r: 4, c: 0, name: 'Y', color: 0xeab308, label: 'Yellow' },
  { r: 4, c: 3, name: 'B', color: 0x2563eb, label: 'Blue' }
];

export default function ThreeTaxiScene({ currentTrace, currentStepIdx, onJumpToStep }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);

  // Mesh refs for animated updates
  const taxiGroupRef = useRef(null);
  const passengerGroupRef = useRef(null);
  const destBeaconRef = useRef(null);
  const pathLinesRef = useRef([]);
  const waypointsRef = useRef([]);

  // Target coordinates for smooth animation
  const animState = useRef({
    targetPos: new THREE.Vector3(0, 0, 0),
    targetRotY: 0,
    currentPos: new THREE.Vector3(0, 0, 0),
    currentRotY: 0
  });

  // Convert (row, col) to 3D world (x, z)
  const cellToWorld = (r, c) => ({
    x: (c - 2) * 2.0,
    z: (r - 2) * 2.0
  });

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // 1. SCENE & CAMERA
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf1f5f9);
    scene.fog = new THREE.FogExp2(0xf1f5f9, 0.025);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(7.5, 9.5, 9.0);
    cameraRef.current = camera;

    // 2. RENDERER
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;
    container.appendChild(renderer.domElement);

    // 3. ORBIT CONTROLS
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2.05;
    controls.minDistance = 4;
    controls.maxDistance = 25;
    controls.target.set(0, 0.2, 0);
    controlsRef.current = controls;

    // 4. LIGHTING
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight.position.set(10, 15, 8);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 30;
    dirLight.shadow.camera.left = -7;
    dirLight.shadow.camera.right = 7;
    dirLight.shadow.camera.top = 7;
    dirLight.shadow.camera.bottom = -7;
    scene.add(dirLight);

    // 5. BASE GROUND & 5x5 TILES
    const baseGeo = new THREE.BoxGeometry(11.2, 0.3, 11.2);
    const baseMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.8 });
    const baseMesh = new THREE.Mesh(baseGeo, baseMat);
    baseMesh.position.y = -0.15;
    baseMesh.receiveShadow = true;
    scene.add(baseMesh);

    // 25 Grid Tile Cells
    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 5; c++) {
        const { x, z } = cellToWorld(r, c);
        const tileGeo = new THREE.BoxGeometry(1.88, 0.04, 1.88);
        const tileMat = new THREE.MeshStandardMaterial({
          color: (r + c) % 2 === 0 ? 0xffffff : 0xf8fafc,
          roughness: 0.6
        });
        const tile = new THREE.Mesh(tileGeo, tileMat);
        tile.position.set(x, 0.02, z);
        tile.receiveShadow = true;
        scene.add(tile);
      }
    }

    // 6. TAXI-v4 INTERNAL BARRIER WALLS
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.5, metalness: 0.2 });
    // Wall 1: between Col 1 & 2 in Row 0-1 (x = -1.0, z from -4 to -2 -> center -3.0)
    const wall1 = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.65, 3.8), wallMat);
    wall1.position.set(-1.0, 0.325, -3.0);
    wall1.castShadow = true;
    wall1.receiveShadow = true;
    scene.add(wall1);

    // Wall 2: between Col 0 & 1 in Row 3-4 (x = -3.0, z from 2 to 4 -> center 3.0)
    const wall2 = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.65, 3.8), wallMat);
    wall2.position.set(-3.0, 0.325, 3.0);
    wall2.castShadow = true;
    wall2.receiveShadow = true;
    scene.add(wall2);

    // Wall 3: between Col 2 & 3 in Row 3-4 (x = 1.0, z from 2 to 4 -> center 3.0)
    const wall3 = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.65, 3.8), wallMat);
    wall3.position.set(1.0, 0.325, 3.0);
    wall3.castShadow = true;
    wall3.receiveShadow = true;
    scene.add(wall3);

    // 7. LANDMARK PADS (R, G, Y, B)
    LANDMARKS.forEach(lm => {
      const { x, z } = cellToWorld(lm.r, lm.c);
      const padGeo = new THREE.CylinderGeometry(0.72, 0.72, 0.06, 32);
      const padMat = new THREE.MeshStandardMaterial({
        color: lm.color,
        roughness: 0.4,
        emissive: lm.color,
        emissiveIntensity: 0.2
      });
      const pad = new THREE.Mesh(padGeo, padMat);
      pad.position.set(x, 0.05, z);
      pad.receiveShadow = true;
      scene.add(pad);
    });

    // 8. DESTINATION PULSING BEACON
    const destGroup = new THREE.Group();
    const beamGeo = new THREE.CylinderGeometry(0.7, 0.7, 2.5, 32, 1, true);
    const beamMat = new THREE.MeshStandardMaterial({
      color: 0x800000,
      transparent: true,
      opacity: 0.35,
      side: THREE.DoubleSide
    });
    const beamMesh = new THREE.Mesh(beamGeo, beamMat);
    beamMesh.position.y = 1.25;
    destGroup.add(beamMesh);

    const ringGeo = new THREE.RingGeometry(0.65, 0.78, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x800000, side: THREE.DoubleSide });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = -Math.PI / 2;
    ringMesh.position.y = 0.08;
    destGroup.add(ringMesh);

    scene.add(destGroup);
    destBeaconRef.current = destGroup;

    // 9. PROCEDURAL 3D TAXI VEHICLE
    const taxiGroup = new THREE.Group();
    // Chassis (Yellow with Maroon roof)
    const bodyGeo = new THREE.BoxGeometry(1.3, 0.42, 0.82);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.3 });
    const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
    bodyMesh.position.y = 0.32;
    bodyMesh.castShadow = true;
    taxiGroup.add(bodyMesh);

    // Maroon racing stripe
    const stripeGeo = new THREE.BoxGeometry(1.32, 0.1, 0.84);
    const stripeMat = new THREE.MeshStandardMaterial({ color: 0x800000 });
    const stripeMesh = new THREE.Mesh(stripeGeo, stripeMat);
    stripeMesh.position.y = 0.3;
    taxiGroup.add(stripeMesh);

    // Cabin
    const cabinGeo = new THREE.BoxGeometry(0.75, 0.34, 0.7);
    const cabinMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.2 });
    const cabinMesh = new THREE.Mesh(cabinGeo, cabinMat);
    cabinMesh.position.set(-0.05, 0.65, 0);
    cabinMesh.castShadow = true;
    taxiGroup.add(cabinMesh);

    // Taxi Roof Sign
    const signGeo = new THREE.BoxGeometry(0.35, 0.12, 0.16);
    const signMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfffbeb, emissiveIntensity: 0.4 });
    const signMesh = new THREE.Mesh(signGeo, signMat);
    signMesh.position.set(-0.05, 0.88, 0);
    taxiGroup.add(signMesh);

    // 4 Wheels
    const wheelGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.12, 16);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.9 });
    const wheelOffsets = [
      [-0.4, 0.18, 0.42], [-0.4, 0.18, -0.42],
      [0.4, 0.18, 0.42], [0.4, 0.18, -0.42]
    ];
    wheelOffsets.forEach(([wx, wy, wz]) => {
      const wheel = new THREE.Mesh(wheelGeo, wheelMat);
      wheel.rotation.x = Math.PI / 2;
      wheel.position.set(wx, wy, wz);
      wheel.castShadow = true;
      taxiGroup.add(wheel);
    });

    // Headlights
    const lightGeo = new THREE.SphereGeometry(0.08, 12, 12);
    const lightMat = new THREE.MeshBasicMaterial({ color: 0xfef08a });
    const lightL = new THREE.Mesh(lightGeo, lightMat);
    lightL.position.set(0.66, 0.34, 0.25);
    const lightR = new THREE.Mesh(lightGeo, lightMat);
    lightR.position.set(0.66, 0.34, -0.25);
    taxiGroup.add(lightL);
    taxiGroup.add(lightR);

    scene.add(taxiGroup);
    taxiGroupRef.current = taxiGroup;

    // 10. PASSENGER 3D CHARACTER
    const passGroup = new THREE.Group();
    // Body
    const pBodyGeo = new THREE.CylinderGeometry(0.16, 0.2, 0.55, 16);
    const pBodyMat = new THREE.MeshStandardMaterial({ color: 0x2563eb });
    const pBody = new THREE.Mesh(pBodyGeo, pBodyMat);
    pBody.position.y = 0.35;
    pBody.castShadow = true;
    passGroup.add(pBody);
    // Head
    const pHeadGeo = new THREE.SphereGeometry(0.16, 16, 16);
    const pHeadMat = new THREE.MeshStandardMaterial({ color: 0xfed7aa });
    const pHead = new THREE.Mesh(pHeadGeo, pHeadMat);
    pHead.position.y = 0.72;
    pHead.castShadow = true;
    passGroup.add(pHead);
    // Cap
    const pCapGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.08, 16);
    const pCapMat = new THREE.MeshStandardMaterial({ color: 0x800000 });
    const pCap = new THREE.Mesh(pCapGeo, pCapMat);
    pCap.position.y = 0.82;
    passGroup.add(pCap);

    scene.add(passGroup);
    passengerGroupRef.current = passGroup;

    // 11. ANIMATION & RENDER LOOP
    let animationFrameId;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      // Smooth taxi movement lerp
      if (taxiGroupRef.current) {
        taxiGroupRef.current.position.lerp(animState.current.targetPos, 0.18);
        // Smooth rotation slerp
        const currentRot = taxiGroupRef.current.rotation.y;
        const targetRot = animState.current.targetRotY;
        taxiGroupRef.current.rotation.y += (targetRot - currentRot) * 0.2;
      }

      // Pulsing destination beacon
      if (destBeaconRef.current) {
        beamMesh.rotation.y += 0.015;
        beamMat.opacity = 0.25 + 0.15 * Math.sin(Date.now() * 0.005);
      }

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize Handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      if (renderer.domElement && container) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Update scene when currentTrace or currentStepIdx changes
  useEffect(() => {
    if (!currentTrace || currentTrace.length === 0 || !sceneRef.current) return;
    const step = currentTrace[currentStepIdx] || currentTrace[0];
    const scene = sceneRef.current;

    // 1. Update Taxi Position & Heading
    const { x, z } = cellToWorld(step.taxi_row, step.taxi_col);
    animState.current.targetPos.set(x, 0, z);

    // Compute heading direction from action
    // 0: South (+z), 1: North (-z), 2: East (+x), 3: West (-x)
    if (step.action === 0) animState.current.targetRotY = -Math.PI / 2; // South
    else if (step.action === 1) animState.current.targetRotY = Math.PI / 2; // North
    else if (step.action === 2) animState.current.targetRotY = 0; // East
    else if (step.action === 3) animState.current.targetRotY = Math.PI; // West

    // 2. Update Passenger Position / Boarding
    if (passengerGroupRef.current) {
      if (step.pass_loc === 4) {
        // Passenger is inside taxi
        passengerGroupRef.current.visible = false;
      } else {
        passengerGroupRef.current.visible = true;
        const pLm = LANDMARKS[step.pass_loc];
        if (pLm) {
          const pw = cellToWorld(pLm.r, pLm.c);
          passengerGroupRef.current.position.set(pw.x, 0, pw.z);
        }
      }
    }

    // 3. Update Destination Beacon
    if (destBeaconRef.current) {
      const dLm = LANDMARKS[step.dest_loc];
      if (dLm) {
        const dw = cellToWorld(dLm.r, dLm.c);
        destBeaconRef.current.position.set(dw.x, 0, dw.z);
      }
    }

    // 4. Update 3D Shortest Path Ribbons & Waypoint Spheres
    // Remove old path lines & waypoints
    pathLinesRef.current.forEach(mesh => scene.remove(mesh));
    pathLinesRef.current = [];
    waypointsRef.current.forEach(mesh => scene.remove(mesh));
    waypointsRef.current = [];

    const fullPoints = currentTrace.map(s => {
      const w = cellToWorld(s.taxi_row, s.taxi_col);
      return new THREE.Vector3(w.x, 0.12, w.z);
    });

    // Draw planned path (dashed line)
    const fullGeo = new THREE.BufferGeometry().setFromPoints(fullPoints);
    const fullMat = new THREE.LineDashedMaterial({
      color: 0x800000,
      dashSize: 0.2,
      gapSize: 0.15,
      opacity: 0.3,
      transparent: true
    });
    const fullLine = new THREE.Line(fullGeo, fullMat);
    fullLine.computeLineDistances();
    scene.add(fullLine);
    pathLinesRef.current.push(fullLine);

    // Draw traveled path (solid vibrant maroon tube)
    if (currentStepIdx > 0) {
      const traveledPoints = fullPoints.slice(0, currentStepIdx + 1);
      const traveledGeo = new THREE.BufferGeometry().setFromPoints(traveledPoints);
      const traveledMat = new THREE.LineBasicMaterial({
        color: 0x800000,
        linewidth: 3
      });
      const traveledLine = new THREE.Line(traveledGeo, traveledMat);
      scene.add(traveledLine);
      pathLinesRef.current.push(traveledLine);
    }

    // Waypoint nodes
    fullPoints.forEach((pt, i) => {
      const isTraveled = (i <= currentStepIdx);
      const isCurrent = (i === currentStepIdx);
      const radius = isCurrent ? 0.18 : (isTraveled ? 0.12 : 0.08);
      const color = isCurrent ? 0xf59e0b : (isTraveled ? 0x800000 : 0x94a3b8);

      const wpGeo = new THREE.SphereGeometry(radius, 12, 12);
      const wpMat = new THREE.MeshStandardMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: isCurrent ? 0.6 : 0.2
      });
      const wpMesh = new THREE.Mesh(wpGeo, wpMat);
      wpMesh.position.copy(pt);
      scene.add(wpMesh);
      waypointsRef.current.push(wpMesh);
    });

  }, [currentTrace, currentStepIdx]);

  // Camera presets
  const setCameraView = (type) => {
    if (!cameraRef.current || !controlsRef.current) return;
    const cam = cameraRef.current;
    const controls = controlsRef.current;
    if (type === 'iso') {
      cam.position.set(7.5, 9.5, 9.0);
      controls.target.set(0, 0.2, 0);
    } else if (type === 'top') {
      cam.position.set(0, 14, 0.01);
      controls.target.set(0, 0, 0);
    } else if (type === 'close') {
      const pos = animState.current.targetPos;
      cam.position.set(pos.x + 3.5, 3.5, pos.z + 3.5);
      controls.target.copy(pos);
    }
    controls.update();
  };

  return (
    <div className="three-canvas-container" ref={mountRef}>
      {/* 3D Viewport Controls Overlay */}
      <div className="three-controls-overlay">
        <button className="three-view-btn" onClick={() => setCameraView('iso')}>
          <FaCube style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Isometric
        </button>
        <button className="three-view-btn" onClick={() => setCameraView('top')}>
          <FaCompass style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Top-Down
        </button>
        <button className="three-view-btn" onClick={() => setCameraView('close')}>
          <FaCamera style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Chase Cam
        </button>
      </div>

      {/* 3D Legend */}
      <div className="three-legend">
        <span style={{ color: '#dc2626' }}>● R</span>
        <span style={{ color: '#16a34a' }}>● G</span>
        <span style={{ color: '#ca8a04' }}>● Y</span>
        <span style={{ color: '#2563eb' }}>● B</span>
        <span style={{ color: '#800000' }}>─ Shortest Path</span>
      </div>
    </div>
  );
}
