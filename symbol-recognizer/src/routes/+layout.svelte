<!-- src/routes/+layout.svelte -->
<script>
    import TopBar from '../components/TopBar.svelte';
</script>
  
<header>
    <TopBar />
</header>
  
<main>
    <slot></slot>
</main>
  
<footer>
    <p>&copy; 2024 My Website</p>
</footer>
  