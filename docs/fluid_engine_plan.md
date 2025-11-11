
# План дипломної роботи: Engine for Fluid Simulation (Rust + ash + winit)

**Тривалість:** 15 тижнів  
**Мова реалізації:** Rust (мінімум зовнішніх бібліотек: `ash`, `winit`, мінімальні математичні crates)  
**Мета:** Розробити рушій симуляції рідин з фізичною частиною (SPH / grid-based) та графічною частиною (реалістичний рендер води). Порівняти три підходи до паралелізму: CPU (Rust), GPU через Vulkan compute shaders (ash), CUDA. Підготувати експериментальні бенчмарки й висновки.

---

## Короткий огляд архітектури та вибори
- **Фізична модель:** почати з Particle-based (SPH — Smoothed Particle Hydrodynamics). Опціонально реалізувати рівне-решітковий (grid-based) метод як альтернативу (на етапі експериментів).
- **Паралелізація:** 
  1. CPU — чистий Rust (multi-threading: Rayon або власні worker threads на `std::thread` + `crossbeam`/канали).  
  2. GPU (Vulkan compute) — через `ash` (compute pipelines, storage buffers, memory barriers).  
  3. CUDA — окремий бекенд (через `rust-cuda` або виклики в C/C++ через `cc` crate), аналіз продуктивності.
- **Графіка:** рендер частинок як поверхні — методи: Screen-space fluid rendering (depth splatting + blurring), marching cubes на густинному полі (імпліцитне поле з частинок), або combination (particles + normal reconstruction + PBR shader).
- **Інструменти/бібліотеки:** `ash`, `winit`, `nalgebra`/`glam` (для векторної математики), `bytemuck`/`zerocopy` для передачі даних в GPU, мінімум залежностей.

---

## Розподіл на 15 тижнів

### Тиждень 1 — Підготовчий: постановка задачі, дослідження
**Прочитати / подивитись:**
- Основи SPH: Monaghan — "Smoothed Particle Hydrodynamics" (оглядові статті).
- "Real-Time Fluid Dynamics for Games" (Bridson) — розділи про фізику рідин.
- Rust: огляд unsafe, ownership для мультипоточності (Rustonomicon — обрані розділи).
**Завдання:**
- Сформулювати цілі дипломної: вимоги, критерії успіху, список експериментів (продуктивність, точність, візуал).
- Підготувати репозиторій з мінімальною структурою (workspace): `engine_core`, `renderer`, `benchmarks`, `cuda_backend`.
**Результат до кінця тижня:** документ "Specification & Research notes" і базовий репозиторій з `README` та CI skeleton.

---

### Тиждень 2 — Базова фізика: SPH на однопотоковому CPU
**Прочитати:**
- Деталі SPH (smoothing kernels, pressure forces, viscosity models).
- Numeric stability: time integration (Leapfrog, Verlet), CFL condition.
**Завдання:**
- Реалізувати базову SPH (single-threaded) у Rust: структура Particle { pos, vel, density, pressure }.
- Простий інтегратор (explicit), граничні умови (стені).
- Unit-тести для обчислення щільності та сил.
**Результат:** працююча однопотокова симуляція на малому наборі частинок (e.g., 1–5k) та набор тестів.

---

### Тиждень 3 — Оптимізації CPU: прості структури даних
**Прочитати:**
- Spatial hashing / uniform grid / k-d tree для пошуку сусідів.
**Завдання:**
- Реалізувати uniform spatial grid або hashing для пошуку сусідів.
- Порівняти з naïve O(n²) на невеликих тестах (лог продуктивності).
**Результат:** значне зниження часу пошуку сусідів; профайли (baseline).

---

### Тиждень 4 — Мультипотоковість на CPU
**Прочитати:**
- Rust concurrency patterns, `std::thread`, `crossbeam`, `rayon`.
**Завдання:**
- Реалізувати multi-threaded версію симулятора (data-parallel over particles). Підтримати варіант з `rayon` і власним thread-pool.
- Переконатись у безпечній роботі з mutable буферами (avoid data races).
**Результат:** multi-threaded симуляція з масштабуванням на кілька ядер; графіки прискорення (speedup).

---

### Тиждень 5 — Стабільність і поліпшення фізики
**Прочитати:**
- Pressure solvers (WCSPH vs. PCISPH), Position Based Fluids (PBF).
**Завдання:**
- Впровадити поліпшення: стабілізація тиску (e.g., iterative density correction або PBF) або згладжування сил.
- Додати адаптивний timestep або обмеження по CFL.
**Результат:** стабільніші симуляції при більшому кроку інтеграції; приклади з 10k+ частинок.

---

### Тиждень 6 — Базовий рендер: візуалізація частинок
**Прочитати:**
- Screen-space fluid rendering techniques, Gaussian splatting, normal reconstruction.
- Основи Vulkan через `ash`: створення surface, swapchain, буферів.
**Завдання:**
- Підключити `winit` + `ash` для базового рендерингу point sprites / splats.
- Передавати позиції частинок у GPU через storage buffer.
**Результат:** візуалізація частинок у вікні (points/splats), basic camera controls.

---

### Тиждень 7 — Surface reconstruction (screen-space)
**Прочитати:**
- Papers / articles: "Smooth Particles for Rendering" / "Screen-Space Fluid Rendering".
**Завдання:**
- Реалізувати depth splatting → blur → normal reconstruction → PBR shading pipeline.
- Налаштувати параметри ядра для кращої видимості рідини.
**Результат:** реалістичніше відображення рідини (гладка поверхня, освітлення).

---

### Тиждень 8 — Поліпшення графіки: marching cubes / implicit field (опційно)
**Прочитати:**
- Marching Cubes, implicit surfaces from particles (e.g., blobby surfaces).
**Завдання:**
- Згенерувати густинне поле на сітці та застосувати marching cubes (CPU або compute shader).
- Порівняти швидкість та якість з screen-space методом.
**Результат:** приклади рендеру з marching cubes або звіт, чому віддаємо перевагу одному методу.

---

### Тиждень 9 — Підготовка до GPU compute (Vulkan compute)
**Прочитати:**
- Vulkan compute pipelines, descriptor sets, synchronization.
- Передача даних між host ↔ device, memory barriers.
**Завдання:**
- Спроєктувати data layout для compute shader (SSBOs), вирішити packing/alignment.
- Реалізувати мінімальний compute shader, що читає позиції і робить просту операцію (debug).
**Результат:** стабільний pipeline для compute, тестовий shader.

---

### Тиждень 10 — Перенесення фізики на GPU (Vulkan compute)
**Прочитати:**
- GPU SPH implementations (статті / GitHub проєкти для натхнення).
**Завдання:**
- Реалізувати SPH kernel в compute shader: density, forces, integration (на GPU).
- Забезпечити порядок обчислень та memory synchronization між кроками.
**Результат:** працююча GPU-версія симуляції (Vulkan compute), базові бенчмарки vs CPU.

---

### Тиждень 11 — Оптимізації GPU
**Прочитати:**
- GPU memory access patterns, shared/local memory equivalents (Vulkan subgroup / push constants), workgroup sizing.
**Завдання:**
- Оптимізувати compute shader: workgroup layout, reduce memory traffic (coalesced loads), use of local arrays.
- Профайли (timings) на кількох GPU конфігураціях (якщо доступні).
**Результат:** покращена продуктивність GPU-реалізації; графіки порівняння.

---

### Тиждень 12 — CUDA бекенд — реалізація
**Прочитати:**
- Основи CUDA (kernels, memory model, streams), rust-cuda або C/C++ interop.
**Завдання:**
- Підготувати бекенд на CUDA: порт SPH kernels у CUDA, налаштування буферів та синхронізації.
- Бенчмарки vs Vulkan compute і CPU.
**Результат:** працездатна CUDA-версія, первинні заміри продуктивності.

---

### Тиждень 13 — Експерименти та порівняння
**Прочитати:**
- Метрики якості: L2 error, mass conservation, energy drift.
**Завдання:**
- Провести серію експериментів: різні розміри задач, чисельні параметри (kernel radius, timestep), апаратні платформи (CPU cores, GPU types).
- Зібрати метрики: time per step, throughput (particles/sec), точність/стабільність.
**Результат:** набір таблиць/графіків для порівняння трьох підходів.

---

### Тиждень 14 — Додаткові фічі та polishing
**Завдання:**
- Додати UI для налаштування параметрів у runtime (набір слайдерів або простий CLI).
- Полірування рендеру (reflections, refractions, post-processing).
- Переконатися у стабільності всіх бекендів, виправити баги.
**Результат:** demo-сцена з керуванням параметрами, запис відео демо.

---

### Тиждень 15 — Підготовка звіту та презентації
**Завдання:**
- Написати дипломну роботу: вступ, огляд літератури, реалізація (архітектура, API), результати експериментів, аналіз, висновки.
- Підготувати презентацію (10–15 слайдів), записати коротке демо відео.
- Резервні тести та фіналізація репозиторію (readme, build instructions).
**Результат:** фінальний звіт, презентація, демо — готові для захисту.

---

## Рекомендована література й ресурси
- Monaghan, J. J. — "Smoothed Particle Hydrodynamics" (review).  
- Bridson, R. — *Fluid Simulation for Computer Graphics* (book).  
- Müller, M., Charypar, D., Gross, M. — "Particle-based fluid simulation for interactive applications."  
- Rustonomicon — розділи про unsafe, concurrency.  
- Vulkan specification / Vulkan Tutorial; `ash` examples.  
- CUDA Programming Guide (if робитимете CUDA бекенд).  
- Papers on screen-space fluid rendering, marching cubes, PBF.

---

## Практичні поради
- **Версії та експерименти:** тримайте чіткі git-бренчі для кожного бекенду (`cpu`, `vulkan`, `cuda`) та тегуйте контрольні результати для реплікації бенчмарків.  
- **Тестування:** автоматизуйте бенчмарки та логування (CSV), щоб легко будувати графіки.  
- **Математика:** не перевинайдайте колеса — використовуйте `nalgebra`/`glam` для векторів, але реалізуйте ядра SPH самостійно.  
- **Безпека пам'яті:** у GPU-бекендах уважно ставтесь до вирівнювання даних (std430) та пакетування структур. Використовуйте `bytemuck`/`zerocopy`.  
- **Репродуктивність:** фіксуйте random seed (якщо використовуєте стохастику), записуйте версії драйверів і GPU для звіту.

---

## Приклад чек-листу для кінця кожного тижня (коротко)
1. Специфікація, репозиторій.  
2. Однопоточний SPH + тести.  
3. Spatial grid + профайли.  
4. Multi-threaded CPU.  
5. Стабілізація фізики (PBF/pressure fixes).  
6. Базовий рендер (points/splats).  
7. Screen-space surface reconstruction.  
8. Marching cubes (опційно).  
9. Vulkan compute setup.  
10. SPH на Vulkan compute.  
11. GPU оптимізації.  
12. CUDA backend.  
13. Експерименти, порівняння.  
14. UI, полірування, демо.  
15. Звіт, презентація, захист.

---

## Формат звіту (рекомендації)
- Вступ / Мотивація  
- Огляд літератури (SPH, PBF, GPU approaches)  
- Розробка (архітектура, data layout, API)  
- Реалізація (CPU, Vulkan, CUDA)  
- Експерименти (метрики, конфігурації)  
- Результати та аналіз (таблиці, графіки)  
- Обмеження та майбутня робота  
- Висновки  
- Додатки (код, команди для запуску)

---

 Якщо хочеш — можу зараз згенерувати:
- шаблон `README.md` для репозиторію;  
- skeleton проекту з Cargo workspaces;  
- прикладні тестові конфігурації для бенчмарків;  
- або одразу зберегти цей план як markdown-файл.

