
  import * as THREE from 'three';
  import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/jsm/loaders/GLTFLoader.js';
  import { VRMLoaderPlugin, VRMUtils } from 'https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@1/lib/three-vrm.module.js';

  const canvas = document.getElementById('canvas');
  const wrap = document.getElementById('vrm-wrap');

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputEncoding = THREE.sRGBEncoding;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
  camera.position.set(0, 1.3, 3.5);
  camera.lookAt(0, 1.0, 0);

  function resize() {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener('resize', resize);

  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(1, 2, 2);
  scene.add(dir);
  const fill = new THREE.DirectionalLight(0x8888ff, 0.3);
  fill.position.set(-1, 0, -1);
  scene.add(fill);

  let currentVRM = null;
  const loader = new GLTFLoader();
  loader.register(parser => new VRMLoaderPlugin(parser));

  function log(msg) {
    const el = document.getElementById('debug-log');
    el.innerHTML += msg + '<br>';
    el.scrollTop = el.scrollHeight;
  }

  function logBones(bonesList) {
    const el = document.getElementById('bones-list');
    el.innerHTML = bonesList.map(b => '<div style="color:#10b981;">✓ ' + b + '</div>').join('');
  }

  window.loadVRM = function() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.vrm';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      document.getElementById('status').textContent = 'Loading ' + file.name + '...';
      log('Loading: ' + file.name);
      const url = URL.createObjectURL(file);
      loader.load(url,
        (gltf) => {
          const vrm = gltf.userData.vrm;
          if (!vrm) { log('❌ Not a valid VRM file'); return; }
          if (currentVRM) {
            scene.remove(currentVRM.scene);
            VRMUtils.deepDispose(currentVRM.scene);
          }
          currentVRM = vrm;
          VRMUtils.removeUnnecessaryVertices(vrm.scene);
          VRMUtils.combineSkeletons(vrm.scene);

            vrm.scene.traverse((obj) => {
              obj.frustumCulled = false;
            });
          vrm.scene.rotation.y = 0;
          scene.add(vrm.scene);
          document.getElementById('status').textContent = '✅ ' + file.name;
          log('✅ Loaded: ' + file.name);
          
         
          
          log('⏳ Settling model (wait 1s)...');
          setTimeout(() => {
            log('🎬 Starting idle animation...');
            enableIdleAnimation();
          }, 1000);
          
          // Debug: log available bones
          const bones = vrm.humanoid?.humanBones;
          if (bones) {
            const boneNames = Object.keys(bones);
            logBones(boneNames);
            log('📦 VRM has ' + boneNames.length + ' bones');
          }
        },
        (progress) => {
          const pct = Math.round(progress.loaded / (progress.total || 1) * 100);
          document.getElementById('status').textContent = 'Loading... ' + pct + '%';
        },
        (err) => {
          document.getElementById('status').textContent = '❌ Failed';
          log('❌ Error: ' + err.message);
        }
      );
    };
    input.click();
  };

  const fingerBones = [
    'rightIndexProximal', 'rightIndexIntermediate', 'rightIndexDistal',
    'rightMiddleProximal', 'rightMiddleIntermediate', 'rightMiddleDistal',
    'rightRingProximal', 'rightRingIntermediate', 'rightRingDistal',
    'rightLittleProximal', 'rightLittleIntermediate', 'rightLittleDistal',
    'rightThumbProximal', 'rightThumbMetacarpal', 'rightThumbDistal',
    'leftIndexProximal', 'leftIndexIntermediate', 'leftIndexDistal',
    'leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal',
    'leftRingProximal', 'leftRingIntermediate', 'leftRingDistal',
    'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal',
    'leftThumbProximal', 'leftThumbMetacarpal', 'leftThumbDistal',
  ];

  // Map server finger names to VRM bone names
  const fingerBoneMap = {
    'rightIndexProximal': 'rightIndexProximal', 'rightIndexIntermediate': 'rightIndexIntermediate', 'rightIndexDistal': 'rightIndexDistal',
    'rightMiddleProximal': 'rightMiddleProximal', 'rightMiddleIntermediate': 'rightMiddleIntermediate', 'rightMiddleDistal': 'rightMiddleDistal',
    'rightRingProximal': 'rightRingProximal', 'rightRingIntermediate': 'rightRingIntermediate', 'rightRingDistal': 'rightRingDistal',
    'rightLittleProximal': 'rightLittleProximal', 'rightLittleIntermediate': 'rightLittleIntermediate', 'rightLittleDistal': 'rightLittleDistal',
    'rightThumbProximal': 'rightThumbProximal', 'rightThumbMetacarpal': 'rightThumbMetacarpal', 'rightThumbDistal': 'rightThumbDistal',
    'leftIndexProximal': 'leftIndexProximal', 'leftIndexIntermediate': 'leftIndexIntermediate', 'leftIndexDistal': 'leftIndexDistal',
    'leftMiddleProximal': 'leftMiddleProximal', 'leftMiddleIntermediate': 'leftMiddleIntermediate', 'leftMiddleDistal': 'leftMiddleDistal',
    'leftRingProximal': 'leftRingProximal', 'leftRingIntermediate': 'leftRingIntermediate', 'leftRingDistal': 'leftRingDistal',
    'leftLittleProximal': 'leftLittleProximal', 'leftLittleIntermediate': 'leftLittleIntermediate', 'leftLittleDistal': 'leftLittleDistal',
    'leftThumbProximal': 'leftThumbProximal', 'leftThumbMetacarpal': 'leftThumbMetacarpal', 'leftThumbDistal': 'leftThumbDistal',
  };

  const socket = io();
  const tmpQuat = new THREE.Quaternion();
  const leftFingers = [
  'leftIndexProximal',
  'leftIndexIntermediate',
  'leftIndexDistal',
  'leftMiddleProximal',
  'leftMiddleIntermediate',
  'leftMiddleDistal',
  'leftRingProximal',
  'leftRingIntermediate',
  'leftRingDistal',
  'leftLittleProximal',
  'leftLittleIntermediate',
  'leftLittleDistal',
  'leftThumbProximal',
  'leftThumbMetacarpal',
  'leftThumbDistal',
];

const rightFingers = [
  'rightIndexProximal',
  'rightIndexIntermediate',
  'rightIndexDistal',
  'rightMiddleProximal',
  'rightMiddleIntermediate',
  'rightMiddleDistal',
  'rightRingProximal',
  'rightRingIntermediate',
  'rightRingDistal',
  'rightLittleProximal',
  'rightLittleIntermediate',
  'rightLittleDistal',
  'rightThumbProximal',
  'rightThumbMetacarpal',
  'rightThumbDistal',
];

const tmpQuat = new THREE.Quaternion();

function resolveBoneNode(boneName) {
  if (!currentVRM?.humanoid) return null;
  return (
    currentVRM.humanoid.getNormalizedBoneNode(boneName) ||
    currentVRM.humanoid?.humanBones?.[boneName]?.node ||
    null
  );
}

function rotateBone(boneName, quatArray, damp = 0.5, flip = { x: 1, y: 1, z: 1 }) {
  if (!currentVRM?.humanoid || !quatArray) return;

  const bone = resolveBoneNode(boneName);
  if (!bone) {
    log(`⚠️ Bone not found: ${boneName}`);
    return;
  }

  let x, y, z, w;
  if (Array.isArray(quatArray) && quatArray.length === 4) {
    [x, y, z, w] = quatArray;
  } else if (quatArray && typeof quatArray === 'object') {
    ({ x, y, z, w } = quatArray);
  } else {
    return;
  }

  x *= flip.x;
  y *= flip.y;
  z *= flip.z;

  tmpQuat.set(x, y, z, w);
  tmpQuat.normalize();
  bone.quaternion.slerp(tmpQuat, damp);
}

socket.on('pose', (data) => {
  // Disable idle animation when real pose data arrives
  disableIdleAnimation();

  const hasFace = !!data.blendshapes;
  const hasPose = !!data.RightUpperArm;
  const hasFingers = !!data.rightIndexProximal || !!data.leftIndexProximal;

  document.getElementById('face-badge').className = 'badge ' + (hasFace ? 'badge-on' : 'badge-off');
  document.getElementById('face-badge').textContent = hasFace ? 'ON' : 'OFF';
  document.getElementById('pose-badge').className = 'badge ' + (hasPose ? 'badge-on' : 'badge-off');
  document.getElementById('pose-badge').textContent = hasPose ? 'ON' : 'OFF';
  document.getElementById('finger-badge').className = 'badge ' + (hasFingers ? 'badge-on' : 'badge-off');
  document.getElementById('finger-badge').textContent = hasFingers ? 'ON' : 'OFF';

  if (data.blendshapes) {
    document.getElementById('mouth-val').textContent = (data.blendshapes.A || 0).toFixed(2);
    document.getElementById('blink-val').textContent =
      (data.blendshapes.Blink_L || 0).toFixed(2) + ' / ' + (data.blendshapes.Blink_R || 0).toFixed(2);
  }
  if (data.RightUpperArm) {
    const [x, y] = data.RightUpperArm;
    document.getElementById('rarm-val').textContent = x.toFixed(2) + ', ' + y.toFixed(2);
  }
  if (data.LeftUpperArm) {
    const [x, y] = data.LeftUpperArm;
    document.getElementById('larm-val').textContent = x.toFixed(2) + ', ' + y.toFixed(2);
  }

  if (!currentVRM || !currentVRM.humanoid) return;

  rotateBone('leftUpperArm', data.LeftUpperArm, 0.35);
  rotateBone('leftLowerArm', data.LeftLowerArm, 0.35);
  rotateBone('rightUpperArm', data.RightUpperArm, 0.35);
  rotateBone('rightLowerArm', data.RightLowerArm, 0.35);
  rotateBone('neck', data.Head, 0.25, { x: 0.7, y: 0.7, z: 0.7 });

  leftFingers.forEach((bone) => {
    if (data[bone]) {
      rotateBone(bone, data[bone], 0.45);
    }
  });

  rightFingers.forEach((bone) => {
    if (data[bone]) {
      rotateBone(bone, data[bone], 0.45);
    }
  });

  if (data.blendshapes && currentVRM.expressionManager) {
    const em = currentVRM.expressionManager;
    try { em.setValue('aa', data.blendshapes.A || 0); } catch (e) {}
    try { em.setValue('blinkLeft', data.blendshapes.Blink_L || 0); } catch (e) {}
    try { em.setValue('blinkRight', data.blendshapes.Blink_R || 0); } catch (e) {}
  }
});

socket.on('translation', (data) => {
  document.getElementById('translation').textContent = data.text;
});

  const clock = new THREE.Clock();
  let idleEnabled = true;
  
  function applyIdleAnimation(vrm, time) {
    if (!vrm || !idleEnabled) return;
    
    const bones = vrm.humanoid?.humanBones;
    if (!bones) return;
    
    // Gentle sway on spine
    const swayAngle = Math.sin(time * 0.5) * 0.1;
    if (bones.spine) {
      bones.spine.node.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), swayAngle);
      bones.spine.node.updateMatrix();
      bones.spine.node.updateMatrixWorld(true);
    }
    
    // Wave right arm up and down
    const armAngle = Math.sin(time * 0.7) * 0.5;
    if (bones.rightUpperArm) {
      const quat = new THREE.Quaternion();
      quat.setFromAxisAngle(new THREE.Vector3(1, 0, 0), armAngle);
      bones.rightUpperArm.node.quaternion.copy(quat);
      bones.rightUpperArm.node.updateMatrix();
      bones.rightUpperArm.node.updateMatrixWorld(true);
    }
    
    // Head look around
    const headAngle = Math.sin(time * 0.3) * 0.15;
    if (bones.head) {
      const quat = new THREE.Quaternion();
      quat.setFromAxisAngle(new THREE.Vector3(0, 1, 0), headAngle);
      bones.head.node.quaternion.copy(quat);
      bones.head.node.updateMatrix();
      bones.head.node.updateMatrixWorld(true);
    }
  }
  
  function disableIdleAnimation() {
    idleEnabled = false;
    log('⏹️ Idle animation disabled (pose tracking active)');
  }
  
  function enableIdleAnimation() {
    idleEnabled = true;
    log('▶️ Idle animation enabled');
  }
  
  function toggleIdleAnimation() {
    idleEnabled = !idleEnabled;
    log(idleEnabled ? '▶️ Idle ON' : '⏹️ Idle OFF');
  }
  
  window.toggleIdle = toggleIdleAnimation;
  
  function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    const elapsed = clock.getElapsedTime();
    
    if (currentVRM) {
      if (idleEnabled) {
  applyIdleAnimation(currentVRM, elapsed);
}
 currentVRM.scene.updateMatrixWorld(true);     

currentVRM.update(delta);
    }
    renderer.render(scene, camera);
  }
  animate();
