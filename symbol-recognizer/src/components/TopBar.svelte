<script>
    export let githubLink = "https://github.com";
</script>

<nav class="navbar fixed-top navbar-custom">
    <div class="container-fluid">
        <div class="project-div d-flex align-items-center">
            <h1 class="project-name mb-0">Symbol Recognizer</h1>
            <h4 class="version-text mb-0 ms-2 text-muted" style="font-size: 0.8em;">v1.0.0</h4>
        </div>
        
        <a class="github-link" href={githubLink} target="_blank" rel="noopener noreferrer">
            GitHub
        </a>
    </div>
</nav>


<style>
    .navbar-custom {
        background-color: #191528;
        border-bottom: 2px solid #5c162e;
    }

    .project-div {
        padding-left: 1.5px;
    }

    .project-name {
        font-size: 25px;
        font-family: "Open Sans";
        color: #7C162E;
    }

    .github-link {
        font-size: 16px;
        color: #333;
        text-decoration: none;
        padding-right: 5px;
    }

    .github-link:hover {
        text-decoration: underline;
    }
</style>