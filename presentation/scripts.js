const nodes = document.querySelectorAll(".timeline-node");
const events = document.querySelectorAll(".event");
let currentIndex = 0;

// Function to display the target event and update the active node
function showEventAndSetActiveNode(targetIndex) {
  events.forEach((event, index) => {
    event.style.display = index === targetIndex ? "block" : "none";
  });

  nodes[currentIndex].classList.remove("active");
  nodes[targetIndex].classList.add("active");
  currentIndex = targetIndex;
}

// Handle click events on timeline nodes
nodes.forEach((node, index) => {
  node.addEventListener("click", () => {
    showEventAndSetActiveNode(index);
  });
});

// Handle arrow key navigation between events
document.addEventListener("keydown", (e) => {
  if (e.code === "ArrowLeft" || e.code === "ArrowRight") {
    const newIndex =
      e.code === "ArrowLeft"
        ? (currentIndex - 1 + events.length) % events.length
        : (currentIndex + 1) % events.length;

    showEventAndSetActiveNode(newIndex);
  }
});

// Handle click event on the demo node
const demoNode = document.querySelector(".demo-node");
demoNode.addEventListener("click", () => {
  const demoEvent = document.querySelector("#demo");

  events.forEach((event) => {
    event.style.display = event === demoEvent ? "block" : "none";
  });
});

// Show all event sections on page load
document.addEventListener("DOMContentLoaded", () => {
  const eventSections = document.querySelectorAll(".event-section");

  function showAllSections() {
    eventSections.forEach((section) => {
      section.querySelector(".event-content").style.display = "block";
    });
  }

  showAllSections();
});

// Set the first node as active on page load
nodes[0].classList.add("active");
