window.onload = () => {
  // 獲得當前 script 的 URL
  const scripts = document.getElementsByTagName('script');
  const scriptSrc = scripts[scripts.length - 1].src; // 獲取最後一個 script 的 src
  const scriptBase = new URL(scriptSrc).origin; // 提取網域的部份

  // 創建 chat 按鈕
  const chatIcon = document.createElement("button");
  chatIcon.style.position = "fixed";
  chatIcon.style.bottom = "20px";
  chatIcon.style.right = "20px";
  chatIcon.style.width = "36px";
  chatIcon.style.height = "36px";
  chatIcon.style.borderRadius = "50%";
  chatIcon.style.backgroundColor = "background-image";
  chatIcon.style.backgroundImage = "url('/public/avatars/assistant.png')"; // 圖片的 URL
  chatIcon.style.backgroundSize = "contain";
  chatIcon.style.backgroundPosition = "center";
  chatIcon.style.border = "none";
  chatIcon.style.cursor = "pointer";
  chatIcon.style.padding = "0";
  chatIcon.style.boxShadow = "0 6px 10px rgba(0, 0, 0, 0.3)"; // 添加陰影讓按鈕更美觀
  document.body.appendChild(chatIcon);

  // 創建 Modal 元件
  const chatModal = document.createElement("div");
  chatModal.style.display = "none";
  chatModal.style.position = "fixed";
  chatModal.style.top = "0";
  chatModal.style.left = "0";
  chatModal.style.width = "100%";
  chatModal.style.height = "100%";
  chatModal.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
  chatModal.style.justifyContent = "center";
  chatModal.style.alignItems = "center";
  chatModal.style.zIndex = "1000";

  const chatFrame = document.createElement("iframe");
  chatFrame.src = `${scriptBase}?ui_mode=modal`; // 動態設置 iframe 的 source
  chatFrame.style.position = "fixed";
  chatFrame.style.bottom = "20px";
  chatFrame.style.right = "20px";
  chatFrame.style.width = "35%";
  chatFrame.style.height = "70%";
  chatFrame.style.border = "none";
  chatFrame.style.borderRadius = "8px";
  chatFrame.style.background = "white";
  chatFrame.style.boxShadow = "0 4px 10px rgba(0, 0, 0, 0.4), 0 0 20px rgba(0, 0, 0, 0.2), inset 0 0 20px rgba(255, 255, 255, 0.2)"; // 添加漸變陰影
  chatModal.appendChild(chatFrame);

  document.body.appendChild(chatModal);

  // 按鈕點擊事件
  chatIcon.addEventListener("click", () => {
    chatModal.style.display = "flex";
  });

  // 點擊 Modal 背景關閉
  chatModal.addEventListener("click", () => {
    chatModal.style.display = "none";
  });

  // 防止點擊 iframe 關閉 Modeal
  chatFrame.addEventListener("click", (e) => e.stopPropagation());
};
