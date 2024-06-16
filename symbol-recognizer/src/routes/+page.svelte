<script>
	import Canvas from "../components/Canvas.svelte";
	import ProgressBars from "../components/ProgressBars.svelte";
	import Button from "../components/Button.svelte"
</script>

<svelte:head>
	<title>Symbol Recognizer</title>
</svelte:head>

<div class="container">
    <div class="row vh-100">
      	<div class="col d-flex flex-column justify-content-center align-items-center">
			<div>
				<Canvas />
			</div>
			
			<div class="mt-1">
				<Button />
			</div>
			
			<div class="mt-5">
				<ProgressBars />
			</div>
      	</div>
    </div>
</div>

<style>
	:global(body) {
    	background-color: #161821;
    	background-size: cover;
    	background-repeat: no-repeat;
    	background-position: center;

		font-family: "Open Sans"
  	}
</style>
