document.addEventListener("DOMContentLoaded", () => {
  const menu = document.querySelector(".barracuda-mobile-menu");
  const trigger = menu?.querySelector("summary");
  if (!menu || !trigger) return;
  const sync = () => trigger.setAttribute("aria-expanded", String(menu.open));
  menu.addEventListener("toggle", sync);
  sync();
});
