// Local, editable slides. Metrics are read from completed artifacts, never invented.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtimeRoot = path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies');
const modules = process.env.RUNTIME_NODE_MODULES || path.join(runtimeRoot, 'node/node_modules');
process.env.RUNTIME_NODE_MODULES = modules;
const skill = path.join(process.env.USERPROFILE, '.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations');
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(modules, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const previewOnly = process.argv.includes('--preview');
const build = path.join(ROOT, 'work/presentation');
await fs.mkdir(build, { recursive: true });
await fs.mkdir(path.join(ROOT, 'reports'), { recursive: true });
const readJSON = async p => JSON.parse(await fs.readFile(path.join(ROOT, p), 'utf8'));
const maybeJSON = async p => { try { return await readJSON(p); } catch { return null; } };
const csv = async p => {
  const lines = (await fs.readFile(path.join(ROOT, p), 'utf8')).trim().split(/\r?\n/);
  const keys = lines.shift().split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((value, i) => [keys[i], value])));
};
const summary = await csv('artifacts/report_esc50/summary.csv');
const snr = await csv('artifacts/report_esc50/snr.csv');
const pump = await maybeJSON('artifacts/anomaly_pump/metrics.json');
const lof = await maybeJSON('artifacts/anomaly_pump/lof/metrics.json');
const project = await maybeJSON('artifacts/ast_project/metrics.json');
if (!previewOnly && (summary.length !== 3 || !pump || !lof || !project)) throw new Error('All final experiment metrics are required');
const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const FONT = 'Arial';
function text(slide, value, x, y, w, h, size=30, bold=false, color='#243547') {
  const shape = slide.shapes.add({ geometry:'textbox', position:{left:x,top:y,width:w,height:h},
    fill:'none', line:{fill:'none',width:0} });
  shape.text = value;
  shape.text.style = { typeface:FONT, fontSize:size, bold, color };
  return shape;
}
function slide(title, notes='') {
  const s=p.slides.add(); s.background.fill='#F8FAFC';
  text(s,title,72,45,1136,100,44,true,'#174D83');
  s.speakerNotes.text=notes;
  return s;
}
function body(s, lines, size=31) {
  lines.forEach((line,i)=>text(s,line,78,165+i*100,1118,88,size));
}
function chart(s,categories,series, max=1, type='bar') {
  // Slides use four decimal places; full-precision source CSV/JSON remains unchanged.
  series = series.map(item => ({...item, values:item.values.map(value => Number(value.toFixed(4)))}));
  const c=s.charts.add(type,{ position:{left:100,top:160,width:1080,height:420},
    categories, series, hasLegend:series.length>1,
    legend:{position:'bottom',textStyle:{fontSize:24}},
    barOptions:{direction:'column',grouping:'clustered'},
    lineOptions:{smooth:false},
    xAxis:{textStyle:{fontSize:24}},
    yAxis:{min:0,max,majorUnit:max/5,numberFormatCode:'0.00',textStyle:{fontSize:22},
      majorGridlines:{fill:'#D7DFE8',width:1}},
    dataLabels:{showValue:type==='bar' && categories.length<=3,position:'outEnd',textStyle:{fontSize:23}},
  });
  applyPresentationChartFont(c,{fontFamily:FONT});
  return c;
}
{
  const s=slide('Как ИИ слышит мир','Локальная лаборатория звука. Сценарий демонстрации: reports/defense.md.');
  text(s,'Фурье, спектрограмма\nи модели звука',78,200,1100,200,64,true);
  text(s,'Математика и воспроизводимые эксперименты\nна публичных аудиодатасетах',82,470,1050,110,34);
}
{
  const s=slide('Звук как дискретная функция','Теорема отсчётов для сигнала с ограниченным спектром. См. SPEC §5, шаг 1; analyzer/services/sampling.py.');
  body(s,['x[n] = f(nΔt),   Δt = 1/fₛ',
    'При fₛ = 16 000 Гц шаг между отсчётами равен 62,5 мкс.',
    'Частота Найквиста равна 8 кГц. Перед понижением fₛ нужен антиалиасинговый фильтр.',
    'Условие восстановления: fₛ > 2fmax для сигнала с ограниченным спектром.']);
}
{
  const s=slide('Ряд Фурье и ортогональность','Численные интегралы и FFT сопоставлены в тестах. Для меандра bₙ=4/(πn) при нечётных n. SPEC §5, шаги 2–3.');
  body(s,['Sₙ(t) = a₀/2 + Σ [aₖ cos(kωt) + bₖ sin(kωt)]',
    'Коэффициент — скалярное произведение сигнала с базисной гармоникой.',
    'Ортогональность позволяет выделить отдельную частоту.',
    'В приложении видно приближение, ошибку RMSE и эффект Гиббса.']);
}
{
  const s=slide('Спектрограмма STFT','Скриншот локального приложения: синтетический свист, окно Hann 1024, 48 кГц. Это иллюстрация обработки, не тестовые метрики.');
  text(s,'STFT — ДПФ в коротких окнах.\n\nДлинное окно лучше разделяет частоты.\n\nКороткое окно точнее локализует событие во времени.',78,180,550,420,32);
  s.images.add({blob:new Uint8Array(await fs.readFile(path.join(ROOT,'work/ui/application.png'))),contentType:'image/png',alt:'Спектрограмма синтетического свиста в приложении',fit:'contain',position:{left:690,top:150,width:500,height:500}});
}
{
  const s=slide('Классификация звуков','AST: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 . CLAP: https://huggingface.co/laion/clap-htsat-unfused . Ревизии закреплены в коде.');
  body(s,['Проектная AST: 10 взаимоисключающих классов, softmax.',
    'Исходная AST: 527 меток AudioSet, независимые sigmoid.',
    'CLAP сопоставляет аудио с текстовыми описаниями.',
    'Порог неуверенности проектной AST выбирается только на validation.']);
}
{
  const s=slide('Сравнение на ESC-50','Источник метрик: artifacts/report_esc50/summary.csv. https://github.com/karolpiczak/ESC-50 . Один seed 2026, пять фолдов; train/val очищены от пересечений источников.');
  chart(s,summary.map(r=>({wave_cnn:'Волна-CNN',mel_cnn:'Мел-CNN',ast:'AST'})[r.model]),[{name:'Accuracy',values:summary.map(r=>+r.accuracy_mean),fill:'#2458A6'}]);
  text(s,'Среднее по пяти фолдам. AST использует предобучение AudioSet.\nМатрицы ошибок и SD сохранены в полном отчёте.',90,600,1100,90,26);
}
{
  const s=slide('Устойчивость к белому шуму','Источник: artifacts/report_esc50/snr.csv и summary.csv. Шум фиксирован, одинаковые реализации по моделям; параметры после test не менялись.');
  const colors=['#2458A6','#DA8B2A','#28886B'];
  const series=summary.map((row,i)=>({name:({wave_cnn:'Волна-CNN',mel_cnn:'Мел-CNN',ast:'AST'})[row.model],fill:colors[i],line:{fill:colors[i],width:3},values:[+row.accuracy_mean,...[20,10,5,0].map(level=>{
    const values=snr.filter(r=>r.model===row.model&&+r.snr_db===level).map(r=>+r.accuracy); return values.reduce((a,b)=>a+b,0)/values.length;
  })]}));
  chart(s,['Чисто','20 дБ','10 дБ','5 дБ','0 дБ'],series,1,'line');
  text(s,'Accuracy по тестовым фолдам. Рост качества при шуме тоже отражается в отчёте.',90,610,1100,65,26);
}
{
  const s=slide('Аномалии насоса','DCASE2020: https://zenodo.org/records/3678171 . CC BY-NC-SA 4.0. Метрики: artifacts/anomaly_pump/metrics.json и lof/metrics.json.');
  if(pump&&lof){
    chart(s,['ID-CNN','AST + LOF'],[
      {name:'Mean AUC',values:[pump.mean_auc,lof.mean_auc],fill:'#2458A6'},
      {name:'Mean pAUC',values:[pump.mean_pauc,lof.mean_pauc],fill:'#DA8B2A'}]);
    text(s,'Обучение на норме. Порог — 90-й перцентиль нормальной валидации.\npAUC стандартизована на FPR ≤ 0,1.',90,605,1100,90,26);
  } else body(s,['Данные готовы. Обучение и финальная оценка ещё выполняются.',
    'Сравниваются классификатор ID машины и LOF на AST-эмбеддингах.']);
}
{
  const s=slide('Классы проекта без собственных записей','FSD50K: https://zenodo.org/records/4060432 . Только CC0/CC BY. Проектное объединение dev+eval с uploader-disjoint split, не официальный benchmark.');
  if(project){
    const conditions=Object.entries(project.conditions);
    chart(s,conditions.map(([key])=>key==='clean'?'Чисто':key.replace('snr_','')+' дБ'),[
      {name:'Accuracy',values:conditions.map(([,v])=>v.accuracy),fill:'#2458A6'},
      {name:'Macro-F1',values:conditions.map(([,v])=>v.macro_f1),fill:'#28886B'}]);
    text(s,'До порога неуверенности. Тест: 516 из 2468 клипов FSD50K.\nСвист: обучение 6, валидация 2, тест 1. Оценка класса неустойчива.',90,605,1100,90,26);
  } else body(s,['Подготовлено 2468 клипов FSD50K с лицензиями CC0 и CC BY.',
    'Данные разделены по авторам до обучения.',
    'Свист малочисленен: 6 / 2 / 1 в train / validation / test.',
    'Итоговые метрики появятся после завершения обучения.']);
}
{
  const s=slide('Честная оценка и ограничения','Протоколы: AGENTS.md, CODEX.md, training/configs/*.yaml, данные и версии в манифестах каждого запуска.');
  body(s,['Test используется для финальной оценки, параметры и пороги выбраны заранее.',
    'Один seed не даёт оценки разброса между независимыми обучениями.',
    'Перенос на новый микрофон или новую машину не проверен.',
    'Галерея демонстрирует работу интерфейса и не заменяет тестовый набор.']);
}
{
  const s=slide('Демонстрация приложения','Сценарий защиты: reports/defense.md. Все вычисления локальные; ресурсы интерфейса и модели после загрузки работают офлайн.');
  body(s,['Волна и дискретизация. Сборка участка из гармоник.',
    'Коэффициенты Фурье, спектр, удаление полосы и прослушивание.',
    'Спектрограмма и сравнение вероятностей до и после шума.',
    'Насос с известным ID: score, порог, ROC и матрица ошибок.']);
}
{
  const s=slide('Материалы и источники','Ссылки на датасеты, лицензии, авторов и точные revision приведены в model cards и attributions.csv. Публикация на Hugging Face не выполнялась.');
  body(s,['Полный отчёт: reports/experiment_report.md',
    'Воспроизводимость: training/configs, data_manifest, env, TensorBoard.',
    'ESC-50, DCASE 2020 и FSD50K с отдельными условиями использования.',
    'Проектная AST и LOF сохранены локально вместе с метриками и источниками.']);
}
for(let i=0;i<p.slides.items.length;i++){
  const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const draft=path.join(build,previewOnly?'preview.pptx':'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(draft);
if(!previewOnly){
  const stamp=new Date().toISOString().replace(/[:.]/g,'-');
  const final=path.join(ROOT,'reports',`presentation_${stamp}.pptx`);
  const receipt=path.join(build,`validation_${stamp}.json`);
  await finalizePresentation({workspaceDir:ROOT,candidatePath:draft,finalPath:final,
    pythonExecutable:path.join(runtimeRoot,'python/python.exe'),
    integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
    explicitTotalSlideCount:12,requiredNativeChartOwnerSlides:[6,7,8,9],
    materializeLiteralChartWorkbooks:true,
    fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:receipt});
  await fs.writeFile(path.join(ROOT,'artifacts/all_tasks/presentation.json'),JSON.stringify({file:final,receipt,
    rendered_slides:build,visual_review_pending:true},null,2));
  console.log('PPTX exported and structurally validated; final visual review remains:',final);
} else console.log('Private preview:',draft);
