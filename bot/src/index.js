const http = require('node:http');
const {
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
  Client,
  EmbedBuilder,
  Events,
  GatewayIntentBits,
  ModalBuilder,
  PermissionFlagsBits,
  REST,
  Routes,
  SlashCommandBuilder,
  TextInputBuilder,
  TextInputStyle,
} = require('discord.js');

const config = require('./config');

const client = new Client({
  intents: [GatewayIntentBits.Guilds],
});

const MANUAL_MODAL_ID = 'pnn_manual_publish_modal';
const EMBED_LIMITS = {
  title: 256,
  description: 4096,
};
const PNN_EMBED_COLOR = 0x2f80ed;
const OWNED_PNN_SUBCOMMANDS = new Set(['publish']);
const OWNED_MANUAL_ACTIONS = new Set(['publish', 'reject']);
const DEPRECATED_PNN_OPTIONS = new Map([
  ['test', 'Create a sample PNN review draft'],
]);
const PRESERVED_COMMAND_KEYS = [
  'default_member_permissions',
  'dm_permission',
  'nsfw',
  'name_localizations',
  'description_localizations',
  'contexts',
  'integration_types',
];

function safeTrim(text, max) {
  const value = String(text || '').trim();
  if (max <= 0) {
    return '';
  }
  if (value.length <= max) {
    return value;
  }
  if (max <= 3) {
    return '.'.repeat(max);
  }
  return `${value.slice(0, Math.max(0, max - 3)).trimEnd()}...`;
}

function wasTrimmed(original, trimmed) {
  return String(original || '').trim() !== String(trimmed || '').trim();
}

function manualReviewButtons(messageId, disabled = false) {
  return [
    new ActionRowBuilder().addComponents(
      new ButtonBuilder()
        .setCustomId(`pnn_manual_publish:${messageId}`)
        .setLabel('Publish to PNN')
        .setStyle(ButtonStyle.Success)
        .setDisabled(disabled),
      new ButtonBuilder()
        .setCustomId(`pnn_manual_reject:${messageId}`)
        .setLabel('Reject Draft')
        .setStyle(ButtonStyle.Danger)
        .setDisabled(disabled),
    ),
  ];
}

function normalizeSectionHeading(line) {
  return String(line || '')
    .trim()
    .replace(/^[\s#*_`>-]+/, '')
    .replace(/[\s:*_`~-]+$/, '')
    .toLowerCase()
    .replace(/\s+/g, ' ');
}

function sectionKeyForHeading(line) {
  const heading = normalizeSectionHeading(line);
  if (['lead & summary', 'lead and summary', 'lead', 'summary'].includes(heading)) {
    return 'summary';
  }
  if (['main developments', 'main development', 'developments', 'article body', 'body'].includes(heading)) {
    return 'body';
  }
  if (['looking ahead', 'look ahead', 'ahead'].includes(heading)) {
    return 'lookingAhead';
  }
  return null;
}

function cleanArticleText(text) {
  return String(text || '')
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function stripLeadingSectionHeading(text, expectedKey) {
  const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
  const firstContentIndex = lines.findIndex((line) => line.trim() !== '');
  if (firstContentIndex === -1) {
    return '';
  }
  if (sectionKeyForHeading(lines[firstContentIndex]) === expectedKey) {
    lines.splice(firstContentIndex, 1);
  }
  return cleanArticleText(lines.join('\n'));
}

function parseStructuredArticle(text) {
  const buckets = {
    summary: [],
    body: [],
    lookingAhead: [],
    preamble: [],
  };
  let currentKey = null;
  let sectionCount = 0;

  for (const line of String(text || '').replace(/\r\n/g, '\n').split('\n')) {
    const nextKey = sectionKeyForHeading(line);
    if (nextKey) {
      currentKey = nextKey;
      sectionCount += 1;
      continue;
    }
    buckets[currentKey || 'preamble'].push(line);
  }

  return {
    summary: cleanArticleText(buckets.summary.join('\n')),
    body: cleanArticleText(buckets.body.join('\n')),
    lookingAhead: cleanArticleText(buckets.lookingAhead.join('\n')),
    sectionCount,
  };
}

function prepareManualArticle(article) {
  const rawSummary = cleanArticleText(article.summary);
  const rawBody = cleanArticleText(article.body);
  const rawLookingAhead = cleanArticleText(article.lookingAhead);
  const combined = [rawSummary, rawBody, rawLookingAhead].filter(Boolean).join('\n\n');
  const parsed = parseStructuredArticle(combined);
  const useParsedSections = parsed.sectionCount >= 2;

  return {
    headline: cleanArticleText(article.headline) || 'Prosperia News Network',
    summary: (useParsedSections && parsed.summary)
      ? parsed.summary
      : stripLeadingSectionHeading(rawSummary, 'summary'),
    body: (useParsedSections && parsed.body)
      ? parsed.body
      : stripLeadingSectionHeading(rawBody, 'body'),
    lookingAhead: (useParsedSections && parsed.lookingAhead)
      ? parsed.lookingAhead
      : stripLeadingSectionHeading(rawLookingAhead, 'lookingAhead'),
  };
}

function renderManualDescription(sections) {
  return [
    `**Lead & Summary**\n${sections.summary}`,
    `**Main Developments**\n${sections.body}`,
    `**Looking Ahead**\n${sections.lookingAhead}`,
  ].join('\n\n');
}

function fitManualDescription(article) {
  const sections = {
    summary: cleanArticleText(article.summary) || 'No lead and summary provided.',
    body: cleanArticleText(article.body) || 'No main developments provided.',
    lookingAhead: cleanArticleText(article.lookingAhead) || 'No looking-ahead notes provided.',
  };
  let trimmed = false;

  for (let attempt = 0; attempt < 20; attempt += 1) {
    const description = renderManualDescription(sections);
    if (description.length <= EMBED_LIMITS.description) {
      return { description, trimmed };
    }

    const overflow = description.length - EMBED_LIMITS.description;
    const longestKey = Object.keys(sections)
      .sort((left, right) => sections[right].length - sections[left].length)[0];
    const current = sections[longestKey];
    const nextLength = Math.max(0, current.length - overflow - 3);
    if (nextLength >= current.length) {
      break;
    }
    sections[longestKey] = safeTrim(current, nextLength);
    trimmed = true;
  }

  return {
    description: safeTrim(renderManualDescription(sections), EMBED_LIMITS.description),
    trimmed: true,
  };
}

function formatFooterTime(date = new Date()) {
  const options = {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
    timeZone: config.timeZone,
  };
  try {
    return new Intl.DateTimeFormat('en-US', options).format(date);
  } catch {
    return new Intl.DateTimeFormat('en-US', { ...options, timeZone: 'UTC' }).format(date);
  }
}

function pnnFooterText(date = new Date()) {
  return `PNN | Signal Over Noise | Today at ${formatFooterTime(date)}`;
}

function buildManualEmbeds(article, mode = 'review') {
  void mode;
  const prepared = prepareManualArticle(article);
  const headline = safeTrim(prepared.headline, EMBED_LIMITS.title) || 'Prosperia News Network';
  const { description, trimmed: descriptionTrimmed } = fitManualDescription(prepared);
  const articleEmbed = new EmbedBuilder()
    .setColor(PNN_EMBED_COLOR)
    .setAuthor({ name: 'Prosperia News Network' })
    .setTitle(headline)
    .setDescription(description)
    .setFooter({ text: pnnFooterText() });

  const trimmed = descriptionTrimmed || wasTrimmed(prepared.headline, headline);
  return { embeds: [articleEmbed], trimmed };
}

function publishedEmbedsFromReview(message) {
  if (message.embeds.length < 1) {
    throw new Error('Review message does not contain a PNN article embed');
  }
  return message.embeds.slice(0, 1).map((sourceEmbed) =>
    EmbedBuilder.from(sourceEmbed)
      .setFooter({ text: pnnFooterText() }),
  );
}

async function postManualReview(article) {
  const channel = await client.channels.fetch(config.reviewChannelId);
  if (!channel?.isTextBased()) {
    throw new Error('PNN_REVIEW_CHANNEL_ID is not a text channel');
  }

  const { embeds, trimmed } = buildManualEmbeds(article, 'review');
  const message = await channel.send({
    embeds,
    allowedMentions: { parse: [] },
  });

  await message.edit({
    components: manualReviewButtons(message.id),
    allowedMentions: { parse: [] },
  });

  return { message, trimmed };
}

function pnnSlashCommandData() {
  return new SlashCommandBuilder()
    .setName('pnn')
    .setDescription('Prosperia News Network publishing')
    .addSubcommand((subcommand) =>
      subcommand
        .setName('publish')
        .setDescription('Create a PNN article for review'),
    )
    .toJSON();
}

function isDeprecatedPnnOption(option) {
  return DEPRECATED_PNN_OPTIONS.get(option.name) === option.description;
}

function mergePnnCommandData(existingCommand, desiredCommand) {
  if (!existingCommand) {
    return desiredCommand;
  }

  const desiredOptions = desiredCommand.options || [];
  const desiredOptionNames = new Set(desiredOptions.map((option) => option.name));
  const preservedOptions = (existingCommand.options || [])
    .filter((option) => !desiredOptionNames.has(option.name))
    .filter((option) => !isDeprecatedPnnOption(option));

  const mergedCommand = {
    name: desiredCommand.name,
    description: existingCommand.description || desiredCommand.description,
    options: [...desiredOptions, ...preservedOptions],
  };

  for (const key of PRESERVED_COMMAND_KEYS) {
    if (existingCommand[key] !== undefined && existingCommand[key] !== null) {
      mergedCommand[key] = existingCommand[key];
    }
  }

  return mergedCommand;
}

async function registerSlashCommands(readyClient) {
  const desiredCommand = pnnSlashCommandData();
  const applicationId = readyClient.application.id;
  if (config.clientId && config.clientId !== applicationId) {
    console.warn(`Configured CLIENT_ID ${config.clientId} differs from logged-in application ${applicationId}; using logged-in application id.`);
  }

  const rest = new REST({ version: '10' }).setToken(config.discordToken);
  const guildCommandsRoute = Routes.applicationGuildCommands(applicationId, config.guildId);

  console.log('Checking existing guild commands...');
  const existingCommands = await rest.get(guildCommandsRoute);
  const existingPnnCommand = existingCommands.find((item) => item.name === desiredCommand.name);
  const command = mergePnnCommandData(existingPnnCommand, desiredCommand);

  if (existingPnnCommand) {
    console.log('Updating /pnn publish while preserving external /pnn subcommands...');
    await rest.patch(
      Routes.applicationGuildCommand(applicationId, config.guildId, existingPnnCommand.id),
      { body: command },
    );
  } else {
    console.log('Creating /pnn command...');
    await rest.post(guildCommandsRoute, { body: command });
  }

  console.log('PNN manual publishing command ready.');
  console.log('Other guild commands preserved.');
}

function hasEditorPermission(interaction) {
  if (!interaction.inGuild()) {
    return false;
  }
  if (interaction.memberPermissions?.has(PermissionFlagsBits.ManageGuild)) {
    return true;
  }
  const memberRoles = interaction.member?.roles;
  if (memberRoles?.cache) {
    return memberRoles.cache.has(config.editorRoleId);
  }
  if (Array.isArray(memberRoles)) {
    return memberRoles.includes(config.editorRoleId);
  }
  return false;
}

function buildManualPublishModal() {
  return new ModalBuilder()
    .setCustomId(MANUAL_MODAL_ID)
    .setTitle('PNN Publish')
    .addComponents(
      new ActionRowBuilder().addComponents(
        new TextInputBuilder()
          .setCustomId('headline')
          .setLabel('Headline')
          .setPlaceholder('[Weekly Briefing] Prosperia enters a more political week')
          .setStyle(TextInputStyle.Short)
          .setRequired(true)
          .setMaxLength(256),
      ),
      new ActionRowBuilder().addComponents(
        new TextInputBuilder()
          .setCustomId('summary')
          .setLabel('Lead & Summary / full article')
          .setPlaceholder('Paste the lead, or a full article with the three section headings.')
          .setStyle(TextInputStyle.Paragraph)
          .setRequired(true)
          .setMaxLength(4000),
      ),
      new ActionRowBuilder().addComponents(
        new TextInputBuilder()
          .setCustomId('body')
          .setLabel('Main Developments (optional)')
          .setPlaceholder('Leave empty if the full article was pasted above.')
          .setStyle(TextInputStyle.Paragraph)
          .setRequired(false)
          .setMaxLength(4000),
      ),
      new ActionRowBuilder().addComponents(
        new TextInputBuilder()
          .setCustomId('looking_ahead')
          .setLabel('Looking Ahead (optional)')
          .setPlaceholder('Leave empty if the full article was pasted above.')
          .setStyle(TextInputStyle.Paragraph)
          .setRequired(false)
          .setMaxLength(1000),
      ),
    );
}

function getModalText(interaction, customId) {
  try {
    return interaction.fields.getTextInputValue(customId);
  } catch {
    return '';
  }
}

async function handlePnnCommand(interaction) {
  const subcommand = interaction.options.getSubcommand(false);
  if (subcommand === 'publish') {
    await interaction.showModal(buildManualPublishModal());
    return;
  }

  if (!OWNED_PNN_SUBCOMMANDS.has(subcommand)) {
    await interaction.reply({
      content: 'This /pnn subcommand is not handled by the PNN publishing bot.',
      ephemeral: true,
    });
  }
}

async function handleManualPublishModal(interaction) {
  if (!hasEditorPermission(interaction)) {
    await interaction.reply({
      content: 'Only PNN editors or members with Manage Server can submit articles.',
      ephemeral: true,
    });
    return;
  }

  await interaction.deferReply({ ephemeral: true });
  const { message, trimmed } = await postManualReview({
    headline: getModalText(interaction, 'headline'),
    summary: getModalText(interaction, 'summary'),
    body: getModalText(interaction, 'body'),
    lookingAhead: getModalText(interaction, 'looking_ahead'),
  });

  await interaction.editReply(`Manual article sent to review. Review message ID: ${message.id}${trimmed ? '\nSome content was trimmed to fit Discord embed limits.' : ''}`);
}

async function handleManualReviewButton(interaction, action, reviewMessageId) {
  if (!hasEditorPermission(interaction)) {
    await interaction.reply({
      content: 'Only PNN editors or members with Manage Server can do that.',
      ephemeral: true,
    });
    return;
  }
  if (reviewMessageId !== interaction.message.id) {
    await interaction.reply({ content: 'This review button does not belong to this message.', ephemeral: true });
    return;
  }

  await interaction.deferReply({ ephemeral: true });

  if (action === 'publish') {
    const newsChannel = await client.channels.fetch(config.newsChannelId);
    if (!newsChannel?.isTextBased()) {
      throw new Error('PNN_NEWS_CHANNEL_ID is not a text channel');
    }

    await newsChannel.send({
      embeds: publishedEmbedsFromReview(interaction.message),
      allowedMentions: { parse: [] },
    });
    await interaction.message.edit({
      content: `Published by ${interaction.user.username}.`,
      components: manualReviewButtons(interaction.message.id, true),
      allowedMentions: { parse: [] },
    });
    await interaction.editReply('Manual article published to PNN news.');
    return;
  }

  if (action === 'reject') {
    await interaction.message.edit({
      content: `Rejected by ${interaction.user.username}.`,
      components: manualReviewButtons(interaction.message.id, true),
      allowedMentions: { parse: [] },
    });
    await interaction.editReply('Manual article rejected.');
  }
}

client.once(Events.ClientReady, async (readyClient) => {
  console.log(`Prosperia News Network manual bot logged in as ${readyClient.user.tag}`);
  try {
    await registerSlashCommands(readyClient);
  } catch (error) {
    console.error('Failed to register PNN slash commands:', error);
  }
});

client.on(Events.InteractionCreate, async (interaction) => {
  try {
    if (interaction.isModalSubmit()) {
      if (interaction.customId === MANUAL_MODAL_ID) {
        await handleManualPublishModal(interaction);
      }
      return;
    }

    if (interaction.isChatInputCommand()) {
      if (interaction.commandName === 'pnn') {
        await handlePnnCommand(interaction);
      }
      return;
    }

    if (!interaction.isButton() || !interaction.customId.startsWith('pnn_manual_')) {
      return;
    }

    const [actionPart, ...idParts] = interaction.customId.split(':');
    const action = actionPart.replace(/^pnn_manual_/, '');
    if (!OWNED_MANUAL_ACTIONS.has(action)) {
      return;
    }
    const reviewMessageId = idParts.join(':');
    if (!reviewMessageId) {
      await interaction.reply({ content: 'Invalid review message id.', ephemeral: true });
      return;
    }
    await handleManualReviewButton(interaction, action, reviewMessageId);
  } catch (error) {
    console.error('PNN interaction failed:', error);
    const failureMessage = `PNN action failed: ${safeTrim(error.message || error, 1500)}`;
    if (interaction.deferred || interaction.replied) {
      await interaction.editReply(failureMessage).catch(() => {});
    } else {
      await interaction.reply({ content: failureMessage, ephemeral: true }).catch(() => {});
    }
  }
});

http
  .createServer((request, response) => {
    if (request.method === 'GET' && (request.url === '/' || request.url === '/health')) {
      response.writeHead(200, { 'Content-Type': 'text/plain' });
      response.end('PNN manual publishing bot online');
      return;
    }
    response.writeHead(404, { 'Content-Type': 'text/plain' });
    response.end('Not found');
  })
  .listen(config.port, '0.0.0.0', () => {
    console.log(`PNN manual publishing health server listening on ${config.port}`);
  });

client.login(config.discordToken);
