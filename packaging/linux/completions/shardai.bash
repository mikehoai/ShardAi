# Bash completion for shardai CLI
_shardai_completions() {
  local cur prev cmd
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"
  cmd=""
  local i=1
  while [[ $i -lt $COMP_CWORD ]]; do
    local word="${COMP_WORDS[$i]}"
    if [[ "$word" != -* ]]; then
      cmd="$word"
      break
    fi
    ((i++))
  done

  local subcommands="create join serve status stop rooms models pull peers chat network doctor service completion version"
  if [[ -z "$cmd" ]]; then
    if [[ "$cur" == -* ]]; then
      COMPREPLY=( $(compgen -W "--help --version --mode --host --port --offline --online" -- "$cur") )
    else
      COMPREPLY=( $(compgen -W "$subcommands" -- "$cur") )
    fi
    return 0
  fi

  case "$cmd" in
    create)
      COMPREPLY=( $(compgen -W "--model --room --room-key --unlisted --no-unlisted --ram --network --source-type --filename --p2p-port --ssd-offload --no-ssd-offload --offline --online --node-id --host --port --foreground --json --help" -- "$cur") )
      ;;
    join)
      COMPREPLY=( $(compgen -W "--room --model --room-key --ram --network --source-type --filename --p2p-port --ssd-offload --no-ssd-offload --offline --online --node-id --host --port --foreground --json --help" -- "$cur") )
      ;;
    serve)
      COMPREPLY=( $(compgen -W "--host --port --offline --online --background --json --help" -- "$cur") )
      ;;
    status)
      COMPREPLY=( $(compgen -W "--host --port --api-url --show-key --json --help" -- "$cur") )
      ;;
    stop)
      COMPREPLY=( $(compgen -W "--host --port --api-url --shutdown-daemon --json --help" -- "$cur") )
      ;;
    rooms|models|network)
      COMPREPLY=( $(compgen -W "--host --port --api-url --offline --online --json --help" -- "$cur") )
      ;;
    pull)
      COMPREPLY=( $(compgen -W "--model --source-type --filename --ram --host --port --api-url --offline --online --json --help" -- "$cur") )
      ;;
    peers)
      COMPREPLY=( $(compgen -W "--host --port --api-url --json --help" -- "$cur") )
      ;;
    chat)
      COMPREPLY=( $(compgen -W "--prompt --model --system --max-tokens --temperature --stream --no-stream --api-key --host --port --api-url --json --help" -- "$cur") )
      ;;
    doctor)
      COMPREPLY=( $(compgen -W "--host --port --api-url --json --help" -- "$cur") )
      ;;
    service)
      if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "--scope --host --port --tracker-url --offline --online --enable --no-enable --json --help" -- "$cur") )
      else
        COMPREPLY=( $(compgen -W "unit install uninstall status start stop restart logs" -- "$cur") )
      fi
      ;;
    completion)
      COMPREPLY=( $(compgen -W "bash zsh --help" -- "$cur") )
      ;;
    version)
      COMPREPLY=( $(compgen -W "--json --help" -- "$cur") )
      ;;
  esac
}
complete -F _shardai_completions shardai
