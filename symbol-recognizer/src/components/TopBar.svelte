<script>
    export let githubLink = "https://github.com";
</script>

<nav class="navbar fixed-top navbar-custom">
    <div class="container-fluid">
        <div class="project-div d-flex align-items-center">
            <h1 class="project-name mb-1">Symbol Recognizer</h1>
            <h4 class="version-text mb-0 ms-2 mt-2" style="font-size: 0.8em;">v1.0.0</h4>
        </div>
        
        <a class="github-link" href={githubLink} target="_blank" rel="noopener noreferrer">
            <img class="github-img" src="/github-mark-white.svg" alt="GitHub">
        </a>
    </div>
</nav>


<style>
    .navbar-custom {
        background-color: #040d1a;
        border-bottom: 4px solid;
        border-image: linear-gradient(90deg, #F23030, #F28705, #F29F05, #F2CB05) 1;
    }

    .project-div {
        padding-left: 1.5px;
    }

    .project-name {
        font-size: 25px;
        color: #7C162E;
    }

    .version-text {
        font-size: 16px;
        font-weight: bold;
        color: #3C162F;
    }

    .github-img {
        width: 28px;
        padding-bottom: 2px;
    }

    .github-link {
        font-size: 22px;
    }

    .github-link:hover {
        text-decoration: underline;
    }
</style>