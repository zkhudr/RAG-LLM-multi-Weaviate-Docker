// static/tooltip.js
document.addEventListener('DOMContentLoaded', function () {
    // Create a function to add tooltips to labels
    function createTooltip(label, key) {
        // Only proceed if we have content for this key
        if (!tooltipContent[key]) return;

        // Create wrapper
        const wrapper = document.createElement('span');
        wrapper.className = 'tooltip-wrapper';

        // Create icon
        const icon = document.createElement('span');
        icon.className = 'tooltip-icon';
        icon.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M12 16v-4"></path>
        <path d="M12 8h.01"></path>
      </svg>
    `;

        // Create tooltip content
        const tooltip = document.createElement('span');
        tooltip.className = 'tooltip-content';
        tooltip.textContent = tooltipContent[key];

        // Assemble tooltip
        wrapper.appendChild(icon);
        wrapper.appendChild(tooltip);

        // Add tooltip after the label text
        label.appendChild(wrapper);
    }

    // Process all labels with data-tooltip attribute
    document.querySelectorAll('[data-tooltip]').forEach(label => {
        const key = label.getAttribute('data-tooltip');
        createTooltip(label, key);
    });
});
