import * as THREE from 'three';

export function showModel(data) {
    console.log('Отображение...');

    const imagePromises = data.map(file => {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => {
                console.warn(`Не удалось загрузить: ${file.name}`);
                resolve(null);
            };
            img.src = URL.createObjectURL(file);
        });
    });
    
    Promise.all(imagePromises).then(images => {
        const validImages = images.filter(img => img !== null);
        console.log(`Загружено изображений: ${validImages.length}`);

        const oldCanvas = document.querySelector('canvas');
        if (oldCanvas) oldCanvas.remove();
        
        new SliceViewer3D(validImages);
    });
}


class SliceViewer3D {
    constructor(pngImagesArray) {
        this.slices = pngImagesArray;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.model = null;
        
        this.init();
        this.processSlices();
        this.animate();
    }
    
    init() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.z = 30;
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        const container = document.getElementById('model');
        container.appendChild(this.renderer.domElement);
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        
        // Освещение и тени
        const mainLight = new THREE.DirectionalLight(0xffffff, 1);
        mainLight.position.set(1, 1, 1).normalize();
        mainLight.castShadow = true; 
  
        mainLight.shadow.mapSize.width = 1024;
        mainLight.shadow.mapSize.height = 1024;
        this.scene.add(mainLight);
        
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
        fillLight.position.set(-1, -0.5, -1);
        this.scene.add(fillLight);
        
        this.scene.add(new THREE.AmbientLight(0x404040));
        
        this.setupControls();
    }
    
    processSlices() {
        if (this.slices.length === 0) return;
        
        const width = this.slices[0].width || 256;
        const height = this.slices[0].height || 256;
        const depth = this.slices.length;
        
        const voxelData = this.createVoxelData(width, height, depth);
    
        this.createPointCloud(voxelData, width, height, depth);
        this.centerCamera(width, height, depth);
    }
    
    createVoxelData(width, height, depth) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = width;
        canvas.height = height;
        
        const voxelData = [];
        const threshold = 128; 
        const stride = 2;
        
        for (let z = 0; z < depth; z += stride) {
            const slice = this.slices[z];
            
            ctx.clearRect(0, 0, width, height);
            ctx.drawImage(slice, 0, 0);
            const imageData = ctx.getImageData(0, 0, width, height);
            
            for (let y = 0; y < height; y += stride) {
                for (let x = 0; x < width; x += stride) {
                    const idx = (y * width + x) * 4;
                    const r = imageData.data[idx];
                    
                    if (r > threshold) {
                        const zScale = 3.0;
                        voxelData.push(
                            (x - width/2) * 0.1,  
                            (y - height/2) * 0.1,  
                            (z - depth/2) * 0.1 * zScale  
                        );
                    }
                }
            }
        }
        
        return voxelData;
    }
    
    createPointCloud(voxelData, width, height, depth) {
        if (voxelData.length === 0) return;
        
        if (this.model) {
            this.scene.remove(this.model);
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', 
            new THREE.Float32BufferAttribute(voxelData, 3));
        
        // Добавление цветов для глубины
        const colors = [];
        for (let i = 0; i < voxelData.length; i += 3) {
            const z = voxelData[i + 2];
            const normalizedZ = (z + depth * 0.05) / (depth * 0.1);
            const depthFactor = 0.5 + normalizedZ * 0.5;
            
            // Градиент 
            colors.push(
                depthFactor * 0.9,
                depthFactor * 0.7,
                depthFactor * 0.8 
            );
        }
        
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.computeVertexNormals(); 

        const material = new THREE.PointsMaterial({
            vertexColors: true,
            size: 1.5,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.9
        });
        
        this.model = new THREE.Points(geometry, material);
        this.model.castShadow = true;
        this.model.receiveShadow = true;
        
        this.scene.add(this.model);
    }
    
    centerCamera(width, height, depth) {
        const maxDim = Math.max(width, height, depth);
        const distance = maxDim * 0.08; 
        
        this.camera.position.z = distance;
        this.camera.lookAt(0, 0, 0);
    }
    
    setupControls() {
        let isDragging = false;
        let previousPosition = { x: 0, y: 0 };
        
        this.renderer.domElement.addEventListener('mousedown', (e) => {
            isDragging = true;
            previousPosition = { x: e.clientX, y: e.clientY };
        });
        
        this.renderer.domElement.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        this.renderer.domElement.addEventListener('mousemove', (e) => {
            if (!isDragging || !this.model) return;
            
            const deltaX = e.clientX - previousPosition.x;
            const deltaY = e.clientY - previousPosition.y;
            
            this.model.rotation.y += deltaX * 0.01;
            this.model.rotation.x += deltaY * 0.01;
            
            previousPosition = { x: e.clientX, y: e.clientY };
        });
        
        this.renderer.domElement.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.camera.position.z *= 1 + e.deltaY * 0.001;
            this.camera.position.z = Math.max(10, Math.min(100, this.camera.position.z));
        });
        
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }
}