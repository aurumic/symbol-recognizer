<script lang="ts">
    import { onDestroy } from 'svelte';
    import { predictions } from '../lib/stores/predictions';

    type Prediction = {
        character: string;
        probability: number;
    };

    let predictionValues: Prediction[] = [];

    const unsubscribe = predictions.subscribe((values) => {
        predictionValues = values;
    });

    onDestroy(() => {
        unsubscribe();
    });
</script>

<div>
    {#if predictionValues.length > 0}
        {#each predictionValues as { character, probability }}
            <div class="progress-container">
                <div class="progress-label">{character}</div>
                <div class="progress">
                    <div
                        class="progress-bar"
                        role="progressbar"
                        style={`width: ${(probability * 100).toFixed(2)}%`}
                        aria-valuenow={probability * 100}
                        aria-valuemin="0"
                        aria-valuemax="100"
                    ></div>
                </div>
                <div class="progress-percentage">{(probability * 100).toFixed(2)}%</div>
            </div>
        {/each}

    {:else}
        {#each Array(5) as _, i}
            <div class="progress-container placeholder">
                <div class="progress-label">-</div>
                <div class="progress">
                    <div
                        class="progress-bar"
                        role="progressbar"
                        style="width: 0%"
                        aria-valuenow="0"
                        aria-valuemin="0"
                        aria-valuemax="100"
                    ></div>
                </div>
                <div class="progress-percentage">0%</div>
            </div>
        {/each}
    {/if}
</div>

<style>
    .progress-container {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .progress-label {
        width: 20px;
        margin-right: 10px;
        text-align: left;
        color: #ffffff;
    }
    .progress {
        flex-grow: 1;
        margin-right: 10px;
        width: 350px;
        background-color: #040d1a;
    }
    .progress-bar {
        transition: width 0.6s ease;
        background-color: #3C162F;
    }
    .progress-percentage {
        width: 50px;
        text-align: right;
        color: #ffffff;
    }
    .placeholder {
      visibility: hidden;
    }
</style>
