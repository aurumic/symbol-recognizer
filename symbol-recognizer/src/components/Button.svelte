<script>
    import { clearCanvas } from "../lib/stores/clearCanvas";
    import { predictions } from "../lib/stores/predictions";
</script>
  
<button class="btn btn-primary" on:click={() => {clearCanvas.set(true); predictions.set([])}}>
    Clear Canvas
</button>
  
<style>
    .btn-primary {
        background-color: #040d1a;
        border-color: #3C162F;
        border-width: 2px;
        border-radius: 1em;
        font-size: 20px;
    }
    .btn-primary:hover {
        background-color: #3C162F;
    }
</style>
  