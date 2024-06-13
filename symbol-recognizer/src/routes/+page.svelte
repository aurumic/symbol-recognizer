<script>
	import Canvas from "../components/Canvas.svelte";
</script>

<svelte:head>
	<title>Home</title>
	<meta name="description" content="Svelte demo app" />
</svelte:head>

<div class="container">
	<div class="container-fluid canvas-container">
		<Canvas />
	</div>
</div>

<style>
	:global(body) {
    	background-color: #110e1b;
    	background-size: cover;
    	background-repeat: no-repeat;
    	background-position: center;
  	}

	.container {
    	display: flex;
    	flex-direction: column;
    	justify-content: center;
    	align-items: center;
    	min-height: calc(100vh - 56px);
    	padding-top: 56px;
  	}

	.canvas-container {
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
</style>
