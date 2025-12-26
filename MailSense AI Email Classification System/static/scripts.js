// ===========================================================
//   DARK MODE SCRIPT — Final Year Project Enhanced Version
// ===========================================================
(function () {

  const btn = document.getElementById('themeToggle');
  const icon = document.getElementById('themeIcon');
  const body = document.body;
  const STORAGE_KEY = "site-theme";

  // ------------------------------
  // Set Theme Mode
  // ------------------------------
  function setMode(mode, firstLoad = false) {
    if (mode === "dark") {
      body.classList.add("dark");
      icon.className = "bi bi-sun-fill";

      localStorage.setItem(STORAGE_KEY, "dark");
    } 
    else {
      body.classList.remove("dark");
      icon.className = "bi bi-moon-fill";

      localStorage.setItem(STORAGE_KEY, "light");
    }

    // Prevent flash on first load
    if (firstLoad) {
      body.style.visibility = "visible";
    }
  }

  // ------------------------------
  // Detect System Preference (optional)
  // ------------------------------
  function detectSystemPreference() {
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  // ------------------------------
  // INITIAL THEME LOAD
  // ------------------------------
  // Hide page while theme is being set to avoid "flash"
  body.style.visibility = "hidden";

  let savedTheme = localStorage.getItem(STORAGE_KEY);

  // If no stored theme → fallback to system theme
  if (!savedTheme) {
    savedTheme = detectSystemPreference();
    localStorage.setItem(STORAGE_KEY, savedTheme);
  }

  // Apply theme instantly (first load)
  setMode(savedTheme, true);

  // ------------------------------
  // Theme Toggle Button
  // ------------------------------
  btn?.addEventListener("click", () => {
    const current = localStorage.getItem(STORAGE_KEY) || "light";
    setMode(current === "light" ? "dark" : "light");
  });

})();
