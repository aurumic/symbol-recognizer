<script lang="ts">
    import { onMount } from "svelte";
    import { clearCanvas } from "../lib/stores/clearCanvas";
    import { predictions } from "../lib/stores/predictions";

    let canvas: HTMLCanvasElement;
    let context: CanvasRenderingContext2D | null = null;
    let drawing: boolean = false;
    let intervalId: number | null = null;

    let interval = 100;
    let line_width = 0.9;

    const startDrawing = () => { drawing = true; startInterval() };
    const stopDrawing = () => { drawing = false; stopInterval() };

    const draw = (event: MouseEvent) => {
        if (!drawing) return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = Math.floor((event.clientX - rect.left) * scaleX);
        const y = Math.floor((event.clientY - rect.top) * scaleY);

        if (context) {
            context.lineWidth = line_width;
            context.lineCap = 'round';
            context.strokeStyle = 'white';
            context.lineTo(x, y);
            context.stroke();
            context.beginPath();
            context.moveTo(x, y);
        }
    };

    const compressDrawing = () => {
        const compressedCanvas = document.createElement('canvas');
        compressedCanvas.width = 28;
        compressedCanvas.height = 28;
        const offscreenContext = compressedCanvas.getContext('2d');

        if (offscreenContext) {
            offscreenContext.drawImage(canvas, 0, 0, 28, 28);
            return compressedCanvas.toDataURL();
        }
        return '';
    };

    const sendSymbol = async (): Promise<void> => {
        try {
            const canvasToProcess = compressDrawing();

            if (canvasToProcess) {
                const data = {
                    imageURL: canvasToProcess,
                };

                const response = await fetch('http://localhost:5000/upload', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                const responseData = await response.json();
                console.log('response recieved');
                predictions.set(responseData)
            } else {
                console.log("canvas compression error");
            }
        } catch (error) {
            console.error('error:', error);
        }
    };
  
    const startInterval = () => {
        if (intervalId === null) {
            intervalId = setInterval(sendSymbol, interval);
        }
    };

    const stopInterval = () => {
        if (intervalId !== null) {
            clearInterval(intervalId);
            intervalId = null;
            
            sendSymbol()
        }
    };

    const resetCanvas = () => {
        context!.fillStyle = '#000000';
        context!.fillRect(0, 0, canvas.width, canvas.height);
    }

    onMount(() => {
        context = canvas.getContext("2d");

        if (context) {
            resetCanvas()

            canvas.addEventListener('mousedown', () => {
                startDrawing();
                context?.beginPath();
            });
            canvas.addEventListener('mouseup', () => {
                stopDrawing();
                context?.closePath();
            });
            canvas.addEventListener('mousemove', draw);
        }

        return () => {
            canvas.removeEventListener('mousedown', startDrawing);
            canvas.removeEventListener('mouseup', stopDrawing);
            canvas.removeEventListener('mousemove', draw);
            stopInterval();
        };
    });

    $: if ($clearCanvas) {
        resetCanvas();
        clearCanvas.set(false);
    }
</script>

<div class="canvas-container">
    <canvas bind:this={ canvas } id="drawingCanvas" width="28" height="28"></canvas>
</div>

<style>
    .canvas-container {
        width: 512px;
        height: 512px;

        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }

    canvas {
        width: 100%;
        height: 100%;
        image-rendering: pixelated;
        border: 3px solid #3C162F;
        border-radius: 1em;
    }
</style>