// Delegate events because Dash mounts the navigation after DOMContentLoaded.
// Native <details> exposes its expanded state to assistive technology.
document.addEventListener("click", (event) => {
  const menu = document.querySelector(".barracuda-mobile-menu[open]");
  if (menu && (!menu.contains(event.target) || event.target.closest("a"))) {
    menu.open = false;
  }
});

document.addEventListener("keydown", (event) => {
  const menu = document.querySelector(".barracuda-mobile-menu[open]");
  if (event.key === "Escape" && menu) {
    menu.open = false;
    menu.querySelector("summary").focus();
  }
});
