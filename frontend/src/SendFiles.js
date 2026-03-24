import { showModel } from "./Show3DModel";

const API_URL = 'http://localhost:8000';

export function handleFolderSelect(event) {

    let selectedFiles = Array.from(event.target.files);
    console.log(`Выбрано файлов: ${selectedFiles.length}`);

    if (selectedFiles.length === 0) {
        alert('Выберите папку с файлами');
        return
    }

    sendFiles(selectedFiles)
}


async function sendFiles(files) {
    console.log('Отправка файлов на сервер...');

    const dcmFiles = files.filter(file => isDcmFile(file));

    if (dcmFiles.length === 0) {
        alert('В папке нет DICOM файлов (.dcm)!');
        return
    }

    console.log(`Найдено DICOM файлов: ${dcmFiles.length}`);

    try {
        // Загружаем все файлы через /upload/folder
        const formData = new FormData();

        dcmFiles.forEach(file => {
            formData.append('files', file);
        });

        console.log('Загрузка файлов на сервер...');

        const uploadResponse = await fetch(`${API_URL}/upload/folder`, {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            const errorData = await uploadResponse.json();
            console.error('Ошибка загрузки:', errorData);
            throw new Error(errorData.detail?.detail || 'Ошибка загрузки файлов');
        }

        const uploadData = await uploadResponse.json();
        console.log('Файлы загружены:', uploadData);

        const seriesId = uploadData.series_id || 'default';

        // Запускаем сегментацию
        console.log('Запуск сегментации...');
        alert('Файлы загружены. Запускаю сегментацию печени...');

        const segmentResponse = await fetch(
            `${API_URL}/segment/dicom?series_id=${seriesId}&target_size_d=64&target_size_h=128&target_size_w=128`,
            { method: 'POST' }
        );

        if (!segmentResponse.ok) {
            const errorData = await segmentResponse.json();
            console.error('Ошибка сегментации:', errorData);
            throw new Error(errorData.detail?.detail || 'Ошибка сегментации');
        }

        const segmentData = await segmentResponse.json();
        console.log('Сегментация завершена:', segmentData);

        // Загружаем маски срезов
        if (segmentData.slices && segmentData.slices.length > 0) {
            console.log('Загрузка масок срезов...');
            const slices = [];

            for (const sliceInfo of segmentData.slices) {
                const img = new Image();
                img.src = sliceInfo.image;
                await new Promise(resolve => img.onload = resolve);
                slices.push(img);
            }

            if (slices.length > 0) {
                showModel(slices);
                console.log('3D модель построена по маскам сегментации');

                // Показываем статистику
                const stats = segmentData.statistics;
                alert(`Сегментация завершена!\nПечень: ${stats.liver_percentage}% объёма\nВокселей печени: ${stats.liver_voxels.toLocaleString()}`);
            } else {
                throw new Error('Не удалось получить маски');
            }
        } else {
            throw new Error('Нет масок в ответе');
        }

    } catch (error) {
        console.error('Ошибка:', error);
        alert('Ошибка: ' + error.message);
    }

    return
}


function isDcmFile(file) {
    const extension = file.name.toLowerCase().split('.').pop();
    
    const mimeType = file.type;
    return extension === 'dcm' || 
           extension === 'dicom' || 
           mimeType === 'application/dicom' ||
           mimeType === 'image/dicom' ||
           mimeType.includes('dicom');
}