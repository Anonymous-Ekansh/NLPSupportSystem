// Setup file for vitest
import "@testing-library/react";

// Mock scrollIntoView for jsdom
window.HTMLElement.prototype.scrollIntoView = function () {};
